"""
Improved Ridge Regression Model with Enhanced Features
Combines weather features + price-based features for better accuracy
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("IMPROVED RIDGE MODEL - ENHANCED FEATURES + WALK-FORWARD VALIDATION")
print("="*80)

# ============================================================================
# Configuration
# ============================================================================

TRAIN_START = '2022-12-05'
FIRST_TEST = '2024-01-01'
ALPHA_GRID = [0.01, 0.1, 1.0, 10.0, 100.0]
COMMODITIES = ['Brent', 'Henry_Hub', 'Power', 'Copper', 'Corn']

OUTPUT_DIR = Path('cleaned_data/model_artifacts')
OUTPUT_DIR.mkdir(exist_ok=True)

print(f"\n⚙️  Configuration:")
print(f"  Train start: {TRAIN_START}")
print(f"  First test: {FIRST_TEST}")
print(f"  Commodities: {', '.join(COMMODITIES)}")

# ============================================================================
# Load Enhanced Price Features
# ============================================================================

print(f"\n📂 Loading enhanced price features...")
enhanced_df = pd.read_csv('cleaned_data/enhanced_price_features.csv')
enhanced_df['week'] = pd.to_datetime(enhanced_df['week'])
print(f"  ✓ Enhanced features: {len(enhanced_df)} records")

# ============================================================================
# Load Weather/EIA Features
# ============================================================================

print(f"\n📂 Loading weather and EIA features...")

interactions_df = pd.read_csv('cleaned_data/weather_eia_interactions.csv')
interactions_df['date'] = pd.to_datetime(interactions_df['date'])
interactions_df['week'] = interactions_df['date'] - pd.to_timedelta(interactions_df['date'].dt.dayofweek, unit='D')

degree_days_df = pd.read_csv('cleaned_data/degree_days.csv')
degree_days_df['week'] = pd.to_datetime(degree_days_df['week'])

# Map commodity names
commodity_map = {
    'Crude_Oil': 'Brent',
    'Natural_Gas': 'Henry_Hub',
    'Power': 'Power',
    'Copper': 'Copper',
    'Corn': 'Corn'
}

interactions_df['commodity_mapped'] = interactions_df['commodity'].map(commodity_map)

# Aggregate to weekly
feature_cols = [col for col in interactions_df.columns if 
                col not in ['date', 'week', 'region', 'commodity', 'commodity_mapped', 'date_eia']]

interactions_weekly = interactions_df.groupby(['week', 'commodity_mapped'])[feature_cols].mean().reset_index()
interactions_weekly = interactions_weekly.rename(columns={'commodity_mapped': 'commodity'})
interactions_weekly = interactions_weekly[interactions_weekly['commodity'].notna()]

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

print(f"  ✓ Weather/EIA interactions: {len(interactions_weekly)} records")
print(f"  ✓ Degree-days: {len(dd_weekly)} records")

# ============================================================================
# Merge All Features
# ============================================================================

print(f"\n🔗 Merging all feature sets...")

# Start with enhanced price features (includes target)
merged_df = enhanced_df.copy()

# Add weather/EIA features
merged_df = merged_df.merge(interactions_weekly, on=['week', 'commodity'], how='left')
merged_df = merged_df.merge(dd_weekly, on='week', how='left')

# Fill missing weather features (Copper/Corn may not have EIA interactions)
print(f"\n  ℹ️  Filling missing weather features with 0...")
weather_cols = [col for col in merged_df.columns if col in feature_cols or 'hdd' in col or 'cdd' in col]
for col in weather_cols:
    if merged_df[col].isna().sum() > 0:
        merged_df[col] = merged_df[col].fillna(0)

# Remove rows with missing targets or critical features
merged_df = merged_df[merged_df['return_next'].notna()].copy()
print(f"  ✓ Final merged dataset: {len(merged_df)} records")

# ============================================================================
# Define Feature Columns
# ============================================================================

exclude_cols = ['week', 'commodity', 'return_next', 'direction_next']
feature_columns = [col for col in merged_df.columns if col not in exclude_cols]

print(f"\n📊 Total features: {len(feature_columns)}")
print(f"  • Price/technical: ~49 features")
print(f"  • Weather/EIA: ~{len([c for c in feature_columns if c in feature_cols])} features")
print(f"  • Degree-days: 8 features")

# ============================================================================
# Walk-Forward Cross-Validation
# ============================================================================

print(f"\n" + "="*80)
print("WALK-FORWARD CROSS-VALIDATION")
print("="*80)

results_list = []
predictions_list = []
fold_num = 0

train_start_dt = pd.to_datetime(TRAIN_START)
first_test_dt = pd.to_datetime(FIRST_TEST)

for commodity in COMMODITIES:
    print(f"\n{'='*80}")
    print(f"COMMODITY: {commodity.upper()}")
    print(f"{'='*80}")
    
    commodity_df = merged_df[merged_df['commodity'] == commodity].sort_values('week').reset_index(drop=True)
    print(f"  Records: {len(commodity_df)}")
    print(f"  Date range: {commodity_df['week'].min().date()} to {commodity_df['week'].max().date()}")
    
    # Get test dates (weekly intervals starting from FIRST_TEST)
    test_weeks = commodity_df[commodity_df['week'] >= first_test_dt]['week'].unique()
    test_weeks = sorted(test_weeks)
    
    print(f"  Test periods: {len(test_weeks)} weeks")
    
    for test_week in test_weeks:
        fold_num += 1
        
        # Split data
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
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Tune alpha on validation split (last 20% of training)
        val_size = max(1, int(len(X_train_scaled) * 0.2))
        X_train_fit = X_train_scaled[:-val_size]
        y_train_fit = y_train.iloc[:-val_size]
        X_val = X_train_scaled[-val_size:]
        y_val = y_train.iloc[-val_size:]
        
        best_alpha = 1.0
        best_val_mae = float('inf')
        
        for alpha in ALPHA_GRID:
            model = Ridge(alpha=alpha)
            model.fit(X_train_fit, y_train_fit)
            val_pred = model.predict(X_val)
            val_mae = mean_absolute_error(y_val, val_pred)
            
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                best_alpha = alpha
        
        # Retrain on full training set with best alpha
        model = Ridge(alpha=best_alpha)
        model.fit(X_train_scaled, y_train)
        
        # Predict
        y_pred = model.predict(X_test_scaled)
        y_pred_direction = (y_pred > 0).astype(int)
        
        # Metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        direction_accuracy = accuracy_score(y_test_direction, y_pred_direction)
        
        # Store results
        results_list.append({
            'fold': fold_num,
            'commodity': commodity,
            'test_week': test_week,
            'train_size': len(train_data),
            'test_size': len(test_data),
            'best_alpha': best_alpha,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'direction_accuracy': direction_accuracy
        })
        
        # Store predictions
        for idx in test_data.index:
            predictions_list.append({
                'fold': fold_num,
                'commodity': commodity,
                'week': test_data.loc[idx, 'week'],
                'y_true': test_data.loc[idx, 'return_next'],
                'y_pred': y_pred[test_data.index.get_loc(idx)],
                'direction_true': test_data.loc[idx, 'direction_next'],
                'direction_pred': y_pred_direction[test_data.index.get_loc(idx)]
            })
        
        # Progress update every 20 folds
        if fold_num % 20 == 0:
            print(f"    Fold {fold_num}: {test_week.date()} | MAE={mae:.4f} | Acc={direction_accuracy:.2%}")
    
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
    comm_mae = comm_results['mae'].mean()
    comm_acc = comm_results['direction_accuracy'].mean()
    print(f"  {commodity:12s}: MAE={comm_mae:.4f} | Accuracy={comm_acc:.2%}")

# ============================================================================
# Save Results
# ============================================================================

results_file = 'reports/improved_model_results.csv'
predictions_file = 'reports/improved_model_predictions.csv'

Path('reports').mkdir(exist_ok=True)
results_df.to_csv(results_file, index=False)
predictions_df.to_csv(predictions_file, index=False)

print(f"\n💾 Saved:")
print(f"  {results_file}")
print(f"  {predictions_file}")

print("\n" + "="*80)
print("✅ IMPROVED MODEL TRAINING COMPLETE!")
print("="*80)
