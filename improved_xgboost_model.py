"""
Improved XGBoost Model with Enhanced Features
Combines price-based + weather features with Optuna optimization
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import xgboost as xgb
import optuna
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score
import warnings
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

print("="*80)
print("IMPROVED XGBOOST MODEL - ENHANCED FEATURES + OPTUNA")
print("="*80)

# ============================================================================
# Configuration
# ============================================================================

TRAIN_START = '2022-12-05'
FIRST_TEST = '2024-01-01'
N_TRIALS = 15  # Optuna trials per commodity
EARLY_STOPPING_ROUNDS = 20
RANDOM_STATE = 42
COMMODITIES = ['Brent', 'Henry_Hub', 'Power', 'Copper', 'Corn']

OUTPUT_DIR = Path('cleaned_data/xgboost_artifacts')
OUTPUT_DIR.mkdir(exist_ok=True)

print(f"\n⚙️  Configuration:")
print(f"  Optuna trials: {N_TRIALS}")
print(f"  Early stopping: {EARLY_STOPPING_ROUNDS}")

# ============================================================================
# Load Enhanced Features
# ============================================================================

print(f"\n📂 Loading enhanced price features...")
enhanced_df = pd.read_csv('cleaned_data/enhanced_price_features.csv')
enhanced_df['week'] = pd.to_datetime(enhanced_df['week'])

print(f"\n📂 Loading weather/EIA features...")
interactions_df = pd.read_csv('cleaned_data/weather_eia_interactions.csv')
interactions_df['date'] = pd.to_datetime(interactions_df['date'])
interactions_df['week'] = interactions_df['date'] - pd.to_timedelta(interactions_df['date'].dt.dayofweek, unit='D')

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

# Merge all features
merged_df = enhanced_df.copy()
merged_df = merged_df.merge(interactions_weekly, on=['week', 'commodity'], how='left')
merged_df = merged_df.merge(dd_weekly, on='week', how='left')
merged_df = merged_df.fillna(0)
merged_df = merged_df[merged_df['return_next'].notna()].copy()

exclude_cols = ['week', 'commodity', 'return_next', 'direction_next']
feature_columns = [col for col in merged_df.columns if col not in exclude_cols]

print(f"  ✓ Total features: {len(feature_columns)}")
print(f"  ✓ Total records: {len(merged_df)}")

# ============================================================================
# XGBoost Training
# ============================================================================

print(f"\n" + "="*80)
print("XGBOOST TRAINING")
print("="*80)

results_list = []
predictions_list = []

train_start_dt = pd.to_datetime(TRAIN_START)
first_test_dt = pd.to_datetime(FIRST_TEST)

for commodity in COMMODITIES:
    print(f"\n{'='*80}")
    print(f"COMMODITY: {commodity.upper()}")
    print(f"{'='*80}")
    
    commodity_df = merged_df[merged_df['commodity'] == commodity].sort_values('week').reset_index(drop=True)
    print(f"  Records: {len(commodity_df)}")
    
    # Get all training data for hyperparameter optimization
    train_all = commodity_df[
        (commodity_df['week'] >= train_start_dt) &
        (commodity_df['week'] < first_test_dt)
    ]
    
    if len(train_all) < 20:
        print(f"  ⚠️  Insufficient training data, skipping...")
        continue
    
    X_train_all = train_all[feature_columns].fillna(0)
    y_train_all = train_all['return_next']
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_all)
    
    # Hyperparameter optimization with Optuna
    print(f"\n  🔍 Optimizing hyperparameters ({N_TRIALS} trials)...")
    
    def objective(trial):
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'mae',
            'max_depth': trial.suggest_int('max_depth', 2, 6),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
            'early_stopping_rounds': EARLY_STOPPING_ROUNDS,
            'random_state': RANDOM_STATE
        }
        
        # 5-fold CV on training data
        split_size = len(X_train_scaled) // 5
        cv_scores = []
        
        for fold_idx in range(3):  # Use 3 folds for speed
            val_start = fold_idx * split_size
            val_end = val_start + split_size
            
            X_tr = np.concatenate([X_train_scaled[:val_start], X_train_scaled[val_end:]])
            y_tr = pd.concat([y_train_all.iloc[:val_start], y_train_all.iloc[val_end:]])
            X_val = X_train_scaled[val_start:val_end]
            y_val = y_train_all.iloc[val_start:val_end]
            
            model = xgb.XGBRegressor(**params)
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
            
            y_val_pred = model.predict(X_val)
            cv_scores.append(mean_absolute_error(y_val, y_val_pred))
        
        return np.mean(cv_scores)
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
    
    best_params = study.best_params
    best_params['objective'] = 'reg:squarederror'
    best_params['eval_metric'] = 'mae'
    best_params['random_state'] = RANDOM_STATE
    # Note: Remove early_stopping_rounds for final training without eval_set
    
    print(f"  ✓ Best MAE: {study.best_value:.4f}")
    print(f"  ✓ Best params: lr={best_params['learning_rate']:.3f}, depth={best_params['max_depth']}, n_est={best_params['n_estimators']}")
    
    # Train final model on all training data
    print(f"\n  📊 Walk-forward validation...")
    
    test_weeks = commodity_df[commodity_df['week'] >= first_test_dt]['week'].unique()
    test_weeks = sorted(test_weeks)
    
    fold_num = 0
    for test_week in test_weeks:
        fold_num += 1
        
        train_data = commodity_df[
            (commodity_df['week'] >= train_start_dt) &
            (commodity_df['week'] < test_week)
        ]
        
        test_data = commodity_df[commodity_df['week'] == test_week]
        
        if len(train_data) < 20 or len(test_data) == 0:
            continue
        
        X_train = train_data[feature_columns].fillna(0)
        y_train = train_data['return_next']
        y_train_direction = train_data['direction_next']
        
        X_test = test_data[feature_columns].fillna(0)
        y_test = test_data['return_next']
        y_test_direction = test_data['direction_next']
        
        # Scale
        scaler_fold = StandardScaler()
        X_train_scaled = scaler_fold.fit_transform(X_train)
        X_test_scaled = scaler_fold.transform(X_test)
        
        # Train with best params
        model = xgb.XGBRegressor(**best_params)
        model.fit(X_train_scaled, y_train, verbose=False)
        
        # Predict
        y_pred = model.predict(X_test_scaled)
        y_pred_direction = (y_pred > 0).astype(int)
        
        # Metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        direction_accuracy = accuracy_score(y_test_direction, y_pred_direction)
        
        results_list.append({
            'commodity': commodity,
            'test_week': test_week,
            'train_size': len(train_data),
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'direction_accuracy': direction_accuracy
        })
        
        for idx in test_data.index:
            predictions_list.append({
                'commodity': commodity,
                'week': test_data.loc[idx, 'week'],
                'y_true': test_data.loc[idx, 'return_next'],
                'y_pred': y_pred[test_data.index.get_loc(idx)],
                'direction_true': test_data.loc[idx, 'direction_next'],
                'direction_pred': y_pred_direction[test_data.index.get_loc(idx)]
            })
        
        if fold_num % 20 == 0:
            print(f"    Week {test_week.date()}: MAE={mae:.4f} | Acc={direction_accuracy:.2%}")
    
    # Commodity summary
    comm_results = [r for r in results_list if r['commodity'] == commodity]
    avg_mae = np.mean([r['mae'] for r in comm_results])
    avg_acc = np.mean([r['direction_accuracy'] for r in comm_results])
    print(f"\n  {commodity} Summary: MAE={avg_mae:.4f} | Accuracy={avg_acc:.2%}")

# ============================================================================
# Overall Results
# ============================================================================

print(f"\n" + "="*80)
print("OVERALL RESULTS")
print("="*80)

results_df = pd.DataFrame(results_list)
predictions_df = pd.DataFrame(predictions_list)

overall_mae = results_df['mae'].mean()
overall_rmse = results_df['rmse'].mean()
overall_r2 = results_df['r2'].mean()
overall_acc = results_df['direction_accuracy'].mean()

print(f"\nAggregate Metrics:")
print(f"  MAE:              {overall_mae:.4f}")
print(f"  RMSE:             {overall_rmse:.4f}")
print(f"  R²:               {overall_r2:.4f}")
print(f"  Direction Acc:    {overall_acc:.2%}")

print(f"\nBy Commodity:")
for commodity in COMMODITIES:
    comm_results = results_df[results_df['commodity'] == commodity]
    if len(comm_results) > 0:
        comm_mae = comm_results['mae'].mean()
        comm_acc = comm_results['direction_accuracy'].mean()
        print(f"  {commodity:12s}: MAE={comm_mae:.4f} | Accuracy={comm_acc:.2%}")

# ============================================================================
# Save Results
# ============================================================================

results_file = 'reports/improved_xgb_results.csv'
predictions_file = 'reports/improved_xgb_predictions.csv'

Path('reports').mkdir(exist_ok=True)
results_df.to_csv(results_file, index=False)
predictions_df.to_csv(predictions_file, index=False)

print(f"\n💾 Saved:")
print(f"  {results_file}")
print(f"  {predictions_file}")

print("\n" + "="*80)
print("✅ IMPROVED XGBOOST TRAINING COMPLETE!")
print("="*80)
