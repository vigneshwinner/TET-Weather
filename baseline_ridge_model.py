"""
Baseline Ridge Regression Model with Walk-Forward Validation
Predicts next-week commodity returns using weather anomalies, degree-days, EIA deltas, and interactions
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, f1_score, roc_auc_score, average_precision_score,
    classification_report, confusion_matrix
)
from sklearn.calibration import calibration_curve
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("BASELINE RIDGE REGRESSION MODEL - WALK-FORWARD VALIDATION")
print("="*80)

# ============================================================================
# Configuration
# ============================================================================

# Training parameters
TRAIN_START = '2022-12-05'
FIRST_TEST = '2024-01-01'
ALPHA_GRID = [0.01, 0.1, 1.0, 10.0, 100.0]

# Commodities to model
COMMODITIES = ['Brent', 'Henry_Hub', 'Power', 'Copper', 'Corn']

# Output directory
OUTPUT_DIR = Path('cleaned_data/model_artifacts')
OUTPUT_DIR.mkdir(exist_ok=True)

print(f"\n⚙️  Configuration:")
print(f"  Train start: {TRAIN_START}")
print(f"  First test: {FIRST_TEST}")
print(f"  Alpha grid: {ALPHA_GRID}")
print(f"  Commodities: {', '.join(COMMODITIES)}")

# ============================================================================
# Load Price Data and Compute Returns
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
    
    # Compute log returns
    df['return'] = np.log(df['Close'] / df['Close'].shift(1))
    df['return_next'] = df['return'].shift(-1)
    df['direction_next'] = (df['return_next'] > 0).astype(int)
    df['week'] = df['Date'] - pd.to_timedelta(df['Date'].dt.dayofweek.astype('int64'), unit='D')  # type: ignore[attr-defined]
    
    df['commodity'] = commodity
    df = df[['Date', 'week', 'commodity', 'Close', 'return', 'return_next', 'direction_next']].copy()
    
    returns_dfs.append(df)
    print(f"  ✓ {commodity}: {len(df)} records, {df['return_next'].notna().sum()} valid targets")

returns_df = pd.concat(returns_dfs, ignore_index=True)
print(f"\n✓ Total price records: {len(returns_df)}")

# ============================================================================
# Load Feature Data
# ============================================================================

print(f"\n📂 Loading feature data...")

interactions_df = pd.read_csv('cleaned_data/weather_eia_interactions.csv')
interactions_df['date'] = pd.to_datetime(interactions_df['date'])
interactions_df['week'] = interactions_df['date'] - pd.to_timedelta(interactions_df['date'].dt.dayofweek.astype('int64'), unit='D')  # type: ignore[attr-defined]
print(f"  ✓ Interactions: {len(interactions_df)} records")

degree_days_df = pd.read_csv('cleaned_data/degree_days.csv')
degree_days_df['week'] = pd.to_datetime(degree_days_df['week'])
print(f"  ✓ Degree-days: {len(degree_days_df)} records")

# ============================================================================
# Aggregate Features to Weekly by Commodity
# ============================================================================

print(f"\n🔧 Aggregating features to weekly level...")

commodity_map = {
    'Crude_Oil': 'Brent',
    'Natural_Gas': 'Henry_Hub',
    'Power': 'Power',
    'Copper': 'Copper',
    'Corn': 'Corn'
}

interactions_df['commodity_mapped'] = interactions_df['commodity'].map(commodity_map)

feature_cols = [col for col in interactions_df.columns if 
                col not in ['date', 'week', 'region', 'commodity', 'commodity_mapped', 'date_eia']]

interactions_weekly = interactions_df.groupby(['week', 'commodity_mapped'])[feature_cols].mean().reset_index()
interactions_weekly = interactions_weekly.rename(columns={'commodity_mapped': 'commodity'})
interactions_weekly = interactions_weekly[interactions_weekly['commodity'].notna()]
print(f"  ✓ Weekly interactions: {len(interactions_weekly)} records")

dd_weekly = degree_days_df.groupby('week').agg({
    'hdd_weekly_sum': 'mean',
    'cdd_weekly_sum': 'mean',
    'hdd_7day_avg': 'mean',
    'cdd_7day_avg': 'mean',
    'hdd_14day_avg': 'mean',
    'cdd_14day_avg': 'mean',
    'hdd_30day_avg': 'mean',
    'cdd_30day_avg': 'mean'
}).reset_index()
print(f"  ✓ Weekly degree-days: {len(dd_weekly)} records")

# ============================================================================
# Merge All Data
# ============================================================================

print(f"\n🔗 Merging features with targets...")

merged_df = returns_df.copy()
merged_df = merged_df.merge(interactions_weekly, on=['week', 'commodity'], how='left')
merged_df = merged_df.merge(dd_weekly, on='week', how='left')

print(f"  ✓ Merged dataset: {len(merged_df)} records")
print(f"  ✓ Features: {len([c for c in merged_df.columns if c not in ['Date', 'week', 'commodity', 'Close', 'return', 'return_next', 'direction_next']])}")

merged_df = merged_df[merged_df['return_next'].notna()].copy()
print(f"  ✓ Records with valid targets: {len(merged_df)}")

# Handle missing features (Copper has no weather-EIA interactions)
print(f"\n  ℹ️  Handling missing features for commodities without weather-EIA interactions...")
missing_by_commodity = merged_df.groupby('commodity').apply(lambda x: x.isnull().sum().sum())
for commodity, missing_count in missing_by_commodity.items():
    if missing_count > 0:
        print(f"    {commodity}: {missing_count} missing values (will fill with 0)")

merged_df = merged_df.fillna(0)
print(f"  ✓ Final records: {len(merged_df)}")

# ============================================================================
# Define Feature Columns
# ============================================================================

exclude_cols = ['Date', 'week', 'commodity', 'Close', 'return', 'return_next', 'direction_next']
feature_columns = [col for col in merged_df.columns if col not in exclude_cols]

print(f"\n📊 Feature set: {len(feature_columns)} features")

# ============================================================================
# Walk-Forward Cross-Validation
# ============================================================================

print(f"\n" + "="*80)
print("WALK-FORWARD CROSS-VALIDATION")
print("="*80)

results_list = []
predictions_list = []
coefficients_list = []

for commodity in COMMODITIES:
    print(f"\n{'='*80}")
    print(f"COMMODITY: {commodity}")
    print(f"{'='*80}")
    
    commodity_df = merged_df[merged_df['commodity'] == commodity].copy()
    commodity_df = commodity_df.sort_values('Date').reset_index(drop=True)
    
    print(f"\n📊 Data for {commodity}: {len(commodity_df)} records")
    print(f"  Date range: {commodity_df['Date'].min().date()} to {commodity_df['Date'].max().date()}")
    
    weeks = sorted(commodity_df['week'].unique())
    first_test_week = pd.to_datetime(FIRST_TEST) - pd.to_timedelta(pd.to_datetime(FIRST_TEST).dayofweek, unit='D')  # type: ignore[attr-defined]
    test_weeks = [w for w in weeks if w >= first_test_week]
    
    print(f"  Test weeks: {len(test_weeks)}")
    
    if len(test_weeks) == 0:
        print(f"  ⚠️  No test weeks available for {commodity}, skipping...")
        continue
    
    fold_num = 0
    
    for test_week in test_weeks:
        fold_num += 1
        
        train_mask = commodity_df['week'] < test_week
        test_mask = commodity_df['week'] == test_week
        
        train_df = commodity_df[train_mask]
        test_df = commodity_df[test_mask]
        
        if len(train_df) < 20 or len(test_df) == 0:
            continue
        
        X_train = train_df[feature_columns].values
        y_train_reg = train_df['return_next'].values
        y_train_clf = train_df['direction_next'].values
        
        X_test = test_df[feature_columns].values
        y_test_reg = test_df['return_next'].values
        y_test_clf = test_df['direction_next'].values
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Regression Model
        best_alpha_reg = ALPHA_GRID[0]
        best_score = -np.inf
        
        n_train = len(X_train_scaled)
        n_val = max(1, int(n_train * 0.2))
        X_train_sub = X_train_scaled[:-n_val]
        y_train_sub = y_train_reg[:-n_val]
        X_val = X_train_scaled[-n_val:]
        y_val = y_train_reg[-n_val:]
        
        for alpha in ALPHA_GRID:
            model = Ridge(alpha=alpha, random_state=42)
            model.fit(X_train_sub, y_train_sub)
            val_pred = model.predict(X_val)
            val_r2 = r2_score(y_val, val_pred)
            if val_r2 > best_score:
                best_score = val_r2
                best_alpha_reg = alpha
        
        reg_model = Ridge(alpha=best_alpha_reg, random_state=42)
        reg_model.fit(X_train_scaled, y_train_reg)
        y_pred_reg = reg_model.predict(X_test_scaled)
        
        mae = mean_absolute_error(y_test_reg, y_pred_reg)
        rmse = np.sqrt(mean_squared_error(y_test_reg, y_pred_reg))
        r2 = r2_score(y_test_reg, y_pred_reg)
        
        # Classification Model
        C_GRID = [0.01, 0.1, 1.0, 10.0, 100.0]
        best_C = C_GRID[0]
        best_score_clf = -np.inf
        
        y_train_clf_sub = y_train_clf[:-n_val]
        y_val_clf = y_train_clf[-n_val:]
        
        for C in C_GRID:
            clf_model = LogisticRegression(penalty='l2', C=C, random_state=42, max_iter=1000)
            clf_model.fit(X_train_sub, y_train_clf_sub)
            val_pred_clf = clf_model.predict_proba(X_val)[:, 1]
            try:
                val_auc = roc_auc_score(y_val_clf, val_pred_clf)
                if val_auc > best_score_clf:
                    best_score_clf = val_auc
                    best_C = C
            except:
                pass
        
        clf_model = LogisticRegression(penalty='l2', C=best_C, random_state=42, max_iter=1000)
        clf_model.fit(X_train_scaled, y_train_clf)
        
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
        
        dir_hit_from_reg = accuracy_score(y_test_clf, y_pred_dir_from_reg)
        
        results_list.append({
            'commodity': commodity,
            'fold': fold_num,
            'test_week': test_week,
            'train_size': len(train_df),
            'test_size': len(test_df),
            'best_alpha_reg': best_alpha_reg,
            'best_C_clf': best_C,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'accuracy': accuracy,
            'f1': f1,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
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
        
        coefficients_list.append({
            'commodity': commodity,
            'fold': fold_num,
            'test_week': test_week,
            'reg_coefficients': reg_model.coef_,
            'reg_intercept': reg_model.intercept_,
            'clf_coefficients': clf_model.coef_[0],
            'clf_intercept': clf_model.intercept_[0],
            'feature_names': feature_columns,
            'scaler': scaler
        })
        
        if fold_num % 10 == 0:
            print(f"  Fold {fold_num}/{len(test_weeks)}: Week {test_week.date()} | "
                  f"R²={r2:.3f}, MAE={mae:.4f}, Acc={accuracy:.3f}, AUC={roc_auc:.3f}")

# ============================================================================
# Aggregate Results
# ============================================================================

print(f"\n" + "="*80)
print("AGGREGATE RESULTS")
print("="*80)

results_df = pd.DataFrame(results_list)
predictions_df = pd.DataFrame(predictions_list)

results_df.to_csv(OUTPUT_DIR / 'fold_results.csv', index=False)
predictions_df.to_csv(OUTPUT_DIR / 'predictions.csv', index=False)
print(f"\n💾 Saved results:")
print(f"  ✓ {OUTPUT_DIR / 'fold_results.csv'}")
print(f"  ✓ {OUTPUT_DIR / 'predictions.csv'}")

for coef_dict in coefficients_list:
    commodity = coef_dict['commodity']
    fold = coef_dict['fold']
    filename = f"{commodity}_fold{fold}_artifacts.joblib"
    joblib.dump(coef_dict, OUTPUT_DIR / filename)

print(f"  ✓ {len(coefficients_list)} coefficient/scaler artifacts saved")

# ============================================================================
# Summary Statistics
# ============================================================================

print(f"\n📊 Overall Performance Summary:")
print(f"\nRegression Metrics (across all folds):")
print(f"  MAE:  {results_df['mae'].mean():.4f} ± {results_df['mae'].std():.4f}")
print(f"  RMSE: {results_df['rmse'].mean():.4f} ± {results_df['rmse'].std():.4f}")
print(f"  R²:   {results_df['r2'].mean():.4f} ± {results_df['r2'].std():.4f}")

print(f"\nClassification Metrics (across all folds):")
print(f"  Accuracy:     {results_df['accuracy'].mean():.4f} ± {results_df['accuracy'].std():.4f}")
print(f"  F1 Score:     {results_df['f1'].mean():.4f} ± {results_df['f1'].std():.4f}")
print(f"  ROC-AUC:      {results_df['roc_auc'].mean():.4f} ± {results_df['roc_auc'].std():.4f}")
print(f"  PR-AUC:       {results_df['pr_auc'].mean():.4f} ± {results_df['pr_auc'].std():.4f}")
print(f"  Dir Hit (Reg): {results_df['dir_hit_from_reg'].mean():.4f} ± {results_df['dir_hit_from_reg'].std():.4f}")

print(f"\n📊 Performance by Commodity:")
for commodity in COMMODITIES:
    commodity_results = results_df[results_df['commodity'] == commodity]
    if len(commodity_results) == 0:
        continue
    
    print(f"\n{commodity}:")
    print(f"  Folds: {len(commodity_results)}")
    print(f"  MAE:      {commodity_results['mae'].mean():.4f}")
    print(f"  RMSE:     {commodity_results['rmse'].mean():.4f}")
    print(f"  R²:       {commodity_results['r2'].mean():.4f}")
    print(f"  Accuracy: {commodity_results['accuracy'].mean():.4f}")
    print(f"  ROC-AUC:  {commodity_results['roc_auc'].mean():.4f}")

print(f"\n" + "="*80)
print("✅ BASELINE MODEL TRAINING COMPLETE!")
print("="*80 + "\n")
