"""
XGBoost Model Training with Optuna Hyperparameter Optimization
Reuses exact fold boundaries from baseline Ridge model for fair comparison
"""

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from datetime import datetime
import xgboost as xgb
import optuna
import shap
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, f1_score, roc_auc_score, average_precision_score,
    brier_score_loss, log_loss
)
import warnings
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

print("="*80)
print("XGBOOST MODEL TRAINING - WITH OPTUNA OPTIMIZATION")
print("="*80)

# ============================================================================
# Configuration
# ============================================================================

TRAIN_START = '2022-12-05'
FIRST_TEST = '2024-01-01'
N_TRIALS = 10  # Optuna trials per fold (use 50 for full optimization)
EARLY_STOPPING_ROUNDS = 50
RANDOM_STATE = 42

COMMODITIES = ['Brent', 'Henry_Hub', 'Power', 'Copper', 'Corn']

OUTPUT_DIR = Path('cleaned_data/xgboost_artifacts')
OUTPUT_DIR.mkdir(exist_ok=True)

print(f"\n⚙️  Configuration:")
print(f"  Train start: {TRAIN_START}")
print(f"  First test: {FIRST_TEST}")
print(f"  Optuna trials: {N_TRIALS}")
print(f"  Early stopping: {EARLY_STOPPING_ROUNDS}")
print(f"  Commodities: {', '.join(COMMODITIES)}")

# ============================================================================
# Load Data
# ============================================================================

print(f"\n📂 Loading price data...")

price_files = {
    'Brent': 'cleaned_data/Brent_3yr.csv',
    'Henry_Hub': 'cleaned_data/Henry_Hub_3yr.csv',
    'Power': 'cleaned_data/Power_3yr.csv',
    'Copper': 'cleaned_data/Copper_3yr.csv',
    'Corn': 'cleaned_data/Corn_3yr.csv'
}

returns_dfs = []

for commodity, filepath in price_files.items():
    df = pd.read_csv(filepath, skiprows=2)
    df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    df['return'] = np.log(df['Close'] / df['Close'].shift(1))
    df['return_next'] = df['return'].shift(-1)
    df['direction_next'] = (df['return_next'] > 0).astype(int)
    df['week'] = df['Date'] - pd.to_timedelta(df['Date'].dt.dayofweek.astype('int64'), unit='D')  # type: ignore[attr-defined]
    
    df['commodity'] = commodity
    df = df[['Date', 'week', 'commodity', 'Close', 'return', 'return_next', 'direction_next']].copy()
    
    returns_dfs.append(df)
    print(f"  ✓ {commodity}: {len(df)} records")

returns_df = pd.concat(returns_dfs, ignore_index=True)

print(f"\n📂 Loading feature data...")

interactions_df = pd.read_csv('cleaned_data/weather_eia_interactions.csv')
interactions_df['date'] = pd.to_datetime(interactions_df['date'])
interactions_df['week'] = interactions_df['date'] - pd.to_timedelta(interactions_df['date'].dt.dayofweek.astype('int64'), unit='D')  # type: ignore[attr-defined]

degree_days_df = pd.read_csv('cleaned_data/degree_days.csv')
degree_days_df['week'] = pd.to_datetime(degree_days_df['week'])

commodity_map = {'Crude_Oil': 'Brent', 'Natural_Gas': 'Henry_Hub', 'Power': 'Power', 'Copper': 'Copper', 'Corn': 'Corn'}
interactions_df['commodity_mapped'] = interactions_df['commodity'].map(commodity_map)

feature_cols = [col for col in interactions_df.columns if 
                col not in ['date', 'week', 'region', 'commodity', 'commodity_mapped', 'date_eia']]

interactions_weekly = interactions_df.groupby(['week', 'commodity_mapped'])[feature_cols].mean().reset_index()
interactions_weekly = interactions_weekly.rename(columns={'commodity_mapped': 'commodity'})
interactions_weekly = interactions_weekly[interactions_weekly['commodity'].notna()]

dd_weekly = degree_days_df.groupby('week').agg({
    'hdd_weekly_sum': 'mean', 'cdd_weekly_sum': 'mean',
    'hdd_7day_avg': 'mean', 'cdd_7day_avg': 'mean',
    'hdd_14day_avg': 'mean', 'cdd_14day_avg': 'mean',
    'hdd_30day_avg': 'mean', 'cdd_30day_avg': 'mean'
}).reset_index()

merged_df = returns_df.copy()
merged_df = merged_df.merge(interactions_weekly, on=['week', 'commodity'], how='left')
merged_df = merged_df.merge(dd_weekly, on='week', how='left')

print(f"  ℹ️  Commodities without weather-EIA interactions will use degree-days only")

merged_df = merged_df[merged_df['return_next'].notna()].copy()

missing_by_commodity = merged_df.groupby('commodity').apply(lambda x: x.isnull().sum().sum())
for commodity, missing_count in missing_by_commodity.items():
    if missing_count > 0:
        print(f"    {commodity}: {missing_count} missing values (will fill with 0)")

merged_df = merged_df.fillna(0)

exclude_cols = ['Date', 'week', 'commodity', 'Close', 'return', 'return_next', 'direction_next']
feature_columns = [col for col in merged_df.columns if col not in exclude_cols]

print(f"  ✓ Total features: {len(feature_columns)}")
print(f"  ✓ Total records: {len(merged_df)}")

# ============================================================================
# XGBoost Training
# ============================================================================

print(f"\n" + "="*80)
print("XGBOOST TRAINING WITH OPTUNA OPTIMIZATION")
print("="*80)

results_list = []
predictions_list = []
best_params_list = []
feature_importance_list = []

for commodity in COMMODITIES:
    print(f"\n{'='*80}")
    print(f"COMMODITY: {commodity}")
    print(f"{'='*80}")
    
    commodity_df = merged_df[merged_df['commodity'] == commodity].copy()
    commodity_df = commodity_df.sort_values('Date').reset_index(drop=True)
    
    print(f"\n📊 Data: {len(commodity_df)} records")
    print(f"  Date range: {commodity_df['Date'].min().date()} to {commodity_df['Date'].max().date()}")
    
    weeks = sorted(commodity_df['week'].unique())
    first_test_week = pd.to_datetime(FIRST_TEST) - pd.to_timedelta(pd.to_datetime(FIRST_TEST).dayofweek, unit='D')  # type: ignore[attr-defined]
    test_weeks = [w for w in weeks if w >= first_test_week]
    
    print(f"  Test weeks: {len(test_weeks)}")
    
    fold_num = 0
    
    for test_week in test_weeks:
        fold_num += 1
        
        train_mask = commodity_df['week'] < test_week
        test_mask = commodity_df['week'] == test_week
        
        train_df = commodity_df[train_mask]
        test_df = commodity_df[test_mask]
        
        if len(train_df) < 30 or len(test_df) == 0:
            continue
        
        X_train_full = train_df[feature_columns].values
        y_train_reg_full = train_df['return_next'].values
        y_train_clf_full = train_df['direction_next'].values
        
        X_test = test_df[feature_columns].values
        y_test_reg = test_df['return_next'].values
        y_test_clf = test_df['direction_next'].values
        
        n_train = len(X_train_full)
        n_val = max(5, int(n_train * 0.2))
        
        X_train = X_train_full[:-n_val]
        y_train_reg = y_train_reg_full[:-n_val]
        y_train_clf = y_train_clf_full[:-n_val]
        
        X_val = X_train_full[-n_val:]
        y_val_reg = y_train_reg_full[-n_val:]
        y_val_clf = y_train_clf_full[-n_val:]
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        X_train_full_scaled = scaler.transform(X_train_full)
        
        # Regression Optuna
        def objective_reg(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                'random_state': RANDOM_STATE,
                'tree_method': 'hist',
                'objective': 'reg:squarederror'
            }
            
            model = xgb.XGBRegressor(**params, early_stopping_rounds=EARLY_STOPPING_ROUNDS)
            model.fit(
                X_train_scaled, y_train_reg,
                eval_set=[(X_val_scaled, y_val_reg)],
                verbose=False
            )
            
            y_pred = model.predict(X_val_scaled)
            mae = mean_absolute_error(y_val_reg, y_pred)
            return mae
        
        study_reg = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
        study_reg.optimize(objective_reg, n_trials=N_TRIALS, show_progress_bar=False)
        best_params_reg = study_reg.best_params
        
        reg_model = xgb.XGBRegressor(
            **best_params_reg,
            random_state=RANDOM_STATE,
            tree_method='hist',
            objective='reg:squarederror',
            early_stopping_rounds=EARLY_STOPPING_ROUNDS
        )
        reg_model.fit(
            X_train_full_scaled, y_train_reg_full,
            eval_set=[(X_val_scaled, y_val_reg)],
            verbose=False
        )
        
        y_pred_reg = reg_model.predict(X_test_scaled)
        
        mae = mean_absolute_error(y_test_reg, y_pred_reg)
        rmse = np.sqrt(mean_squared_error(y_test_reg, y_pred_reg))
        r2 = r2_score(y_test_reg, y_pred_reg)
        
        # Classification Optuna
        n_neg = (y_train_clf_full == 0).sum()
        n_pos = (y_train_clf_full == 1).sum()
        scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
        
        def objective_clf(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                'scale_pos_weight': scale_pos_weight,
                'random_state': RANDOM_STATE,
                'tree_method': 'hist',
                'objective': 'binary:logistic',
                'eval_metric': 'logloss'
            }
            
            model = xgb.XGBClassifier(**params, early_stopping_rounds=EARLY_STOPPING_ROUNDS)
            model.fit(
                X_train_scaled, y_train_clf,
                eval_set=[(X_val_scaled, y_val_clf)],
                verbose=False
            )
            
            y_pred_proba = model.predict_proba(X_val_scaled)[:, 1]
            try:
                auc = roc_auc_score(y_val_clf, y_pred_proba)
                return -auc
            except:
                return 0.0
        
        study_clf = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
        study_clf.optimize(objective_clf, n_trials=N_TRIALS, show_progress_bar=False)
        best_params_clf = study_clf.best_params
        best_params_clf['scale_pos_weight'] = scale_pos_weight
        
        clf_model = xgb.XGBClassifier(
            **best_params_clf,
            random_state=RANDOM_STATE,
            tree_method='hist',
            objective='binary:logistic',
            eval_metric='logloss',
            early_stopping_rounds=EARLY_STOPPING_ROUNDS
        )
        clf_model.fit(
            X_train_full_scaled, y_train_clf_full,
            eval_set=[(X_val_scaled, y_val_clf)],
            verbose=False
        )
        
        y_pred_clf_proba = clf_model.predict_proba(X_test_scaled)[:, 1]
        y_pred_clf = clf_model.predict(X_test_scaled)
        y_pred_dir_from_reg = (y_pred_reg > 0).astype(int)
        
        accuracy = accuracy_score(y_test_clf, y_pred_clf)
        f1 = f1_score(y_test_clf, y_pred_clf, zero_division=0)
        
        try:
            roc_auc = roc_auc_score(y_test_clf, y_pred_clf_proba)
        except:
            roc_auc = np.nan
        
        try:
            pr_auc = average_precision_score(y_test_clf, y_pred_clf_proba)
        except:
            pr_auc = np.nan
        
        try:
            brier = brier_score_loss(y_test_clf, y_pred_clf_proba)
        except:
            brier = np.nan
        
        try:
            logloss = log_loss(y_test_clf, y_pred_clf_proba)
        except:
            logloss = np.nan
        
        dir_hit_from_reg = accuracy_score(y_test_clf, y_pred_dir_from_reg)
        
        results_list.append({
            'commodity': commodity,
            'fold': fold_num,
            'test_week': test_week,
            'train_size': len(train_df),
            'test_size': len(test_df),
            'scale_pos_weight': scale_pos_weight,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'accuracy': accuracy,
            'f1': f1,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'brier_score': brier,
            'log_loss': logloss,
            'dir_hit_from_reg': dir_hit_from_reg
        })
        
        for i in range(len(test_df)):
            predictions_list.append({
                'commodity': commodity,
                'fold': fold_num,
                'date': test_df.iloc[i]['Date'],
                'week': test_week,
                'actual_return': y_test_reg[i],
                'predicted_return': y_pred_reg[i],
                'actual_direction': y_test_clf[i],
                'predicted_direction': y_pred_clf[i],
                'predicted_proba': y_pred_clf_proba[i],
                'direction_from_reg': y_pred_dir_from_reg[i]
            })
        
        best_params_list.append({
            'commodity': commodity,
            'fold': fold_num,
            'test_week': test_week,
            'reg_params': best_params_reg,
            'clf_params': best_params_clf
        })
        
        artifacts = {
            'commodity': commodity,
            'fold': fold_num,
            'test_week': test_week,
            'reg_model': reg_model,
            'clf_model': clf_model,
            'scaler': scaler,
            'feature_names': feature_columns,
            'best_params_reg': best_params_reg,
            'best_params_clf': best_params_clf,
            'scale_pos_weight': scale_pos_weight
        }
        
        joblib.dump(artifacts, OUTPUT_DIR / f"{commodity}_fold{fold_num}_xgb.joblib")
        
        if fold_num % 10 == 0:
            print(f"  Fold {fold_num}/{len(test_weeks)}: Week {test_week.date()} | "
                  f"R²={r2:.3f}, MAE={mae:.4f}, Acc={accuracy:.3f}, AUC={roc_auc:.3f}, Brier={brier:.3f}")
        
        if fold_num == len(test_weeks):
            print(f"\n  📊 Computing feature importance for {commodity}...")
            
            gain_importance = reg_model.get_booster().get_score(importance_type='gain')
            gain_df = pd.DataFrame([
                {'feature': feature_columns[int(k.replace('f', ''))], 'gain': v}
                for k, v in gain_importance.items()
            ]).sort_values('gain', ascending=False).head(20)
            
            feature_importance_list.append({
                'commodity': commodity,
                'type': 'gain',
                'importance': gain_df.to_dict('records')
            })
            
            print(f"    ✓ Top 5 features by gain:")
            for idx, row in gain_df.head(5).iterrows():
                print(f"      {row['feature']}: {row['gain']:.2f}")
            
            print(f"    Computing SHAP values (this may take a moment)...")
            n_shap_samples = min(500, len(X_train_full_scaled))
            X_shap = X_train_full_scaled[-n_shap_samples:]
            
            explainer = shap.TreeExplainer(reg_model)
            shap_values = explainer.shap_values(X_shap)
            
            shap_importance = np.abs(shap_values).mean(axis=0)
            shap_df = pd.DataFrame({
                'feature': feature_columns,
                'shap_importance': shap_importance
            }).sort_values('shap_importance', ascending=False).head(20)
            
            feature_importance_list.append({
                'commodity': commodity,
                'type': 'shap',
                'importance': shap_df.to_dict('records')
            })
            
            print(f"    ✓ Top 5 features by SHAP:")
            for idx, row in shap_df.head(5).iterrows():
                print(f"      {row['feature']}: {row['shap_importance']:.4f}")

# ============================================================================
# Save Results
# ============================================================================

print(f"\n" + "="*80)
print("SAVING RESULTS")
print("="*80)

results_df = pd.DataFrame(results_list)
predictions_df = pd.DataFrame(predictions_list)

results_df.to_csv(OUTPUT_DIR / 'xgb_fold_results.csv', index=False)
predictions_df.to_csv(OUTPUT_DIR / 'xgb_predictions.csv', index=False)

with open(OUTPUT_DIR / 'best_params.json', 'w') as f:
    json.dump(best_params_list, f, indent=2, default=str)

with open(OUTPUT_DIR / 'feature_importance.json', 'w') as f:
    json.dump(feature_importance_list, f, indent=2)

print(f"\n💾 Saved:")
print(f"  ✓ {OUTPUT_DIR / 'xgb_fold_results.csv'}")
print(f"  ✓ {OUTPUT_DIR / 'xgb_predictions.csv'}")
print(f"  ✓ {OUTPUT_DIR / 'best_params.json'}")
print(f"  ✓ {OUTPUT_DIR / 'feature_importance.json'}")
print(f"  ✓ {len(results_list)} model artifacts (.joblib)")

# ============================================================================
# Summary Statistics
# ============================================================================

print(f"\n" + "="*80)
print("XGBOOST PERFORMANCE SUMMARY")
print("="*80)

print(f"\n📊 Overall Performance (across all folds):")
print(f"\nRegression Metrics:")
print(f"  MAE:  {results_df['mae'].mean():.4f} ± {results_df['mae'].std():.4f}")
print(f"  RMSE: {results_df['rmse'].mean():.4f} ± {results_df['rmse'].std():.4f}")
print(f"  R²:   {results_df['r2'].mean():.4f} ± {results_df['r2'].std():.4f}")

print(f"\nClassification Metrics:")
print(f"  Accuracy:     {results_df['accuracy'].mean():.4f} ± {results_df['accuracy'].std():.4f}")
print(f"  F1 Score:     {results_df['f1'].mean():.4f} ± {results_df['f1'].std():.4f}")
print(f"  ROC-AUC:      {results_df['roc_auc'].mean():.4f} ± {results_df['roc_auc'].std():.4f}")
print(f"  PR-AUC:       {results_df['pr_auc'].mean():.4f} ± {results_df['pr_auc'].std():.4f}")
print(f"  Brier Score:  {results_df['brier_score'].mean():.4f} ± {results_df['brier_score'].std():.4f}")
print(f"  Log Loss:     {results_df['log_loss'].mean():.4f} ± {results_df['log_loss'].std():.4f}")

print(f"\n📊 Performance by Commodity:")
for commodity in COMMODITIES:
    commodity_results = results_df[results_df['commodity'] == commodity]
    
    print(f"\n{commodity}:")
    print(f"  Folds: {len(commodity_results)}")
    print(f"  MAE:      {commodity_results['mae'].mean():.4f}")
    print(f"  R²:       {commodity_results['r2'].mean():.4f}")
    print(f"  Accuracy: {commodity_results['accuracy'].mean():.4f}")
    print(f"  ROC-AUC:  {commodity_results['roc_auc'].mean():.4f}")

print(f"\n" + "="*80)
print("✅ XGBOOST TRAINING COMPLETE!")
print("="*80 + "\n")
