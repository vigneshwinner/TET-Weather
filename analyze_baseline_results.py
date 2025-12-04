"""
Analyze and Visualize Baseline Ridge and XGBoost Model Results
Compare performance across commodities and time
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

print("="*80)
print("BASELINE MODEL RESULTS ANALYSIS")
print("="*80)

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

# Load results
print("\n📂 Loading results...")

baseline_dir = Path('cleaned_data/model_artifacts')
xgb_dir = Path('cleaned_data/xgboost_artifacts')

baseline_results = pd.read_csv(baseline_dir / 'fold_results.csv')
baseline_predictions = pd.read_csv(baseline_dir / 'predictions.csv')
baseline_predictions['date'] = pd.to_datetime(baseline_predictions['date'])

xgb_results = pd.read_csv(xgb_dir / 'xgb_fold_results.csv')
xgb_predictions = pd.read_csv(xgb_dir / 'xgb_predictions.csv')
xgb_predictions['date'] = pd.to_datetime(xgb_predictions['date'])

print(f"  ✓ Baseline: {len(baseline_results)} folds, {len(baseline_predictions)} predictions")
print(f"  ✓ XGBoost: {len(xgb_results)} folds, {len(xgb_predictions)} predictions")

# Create output directory for plots
output_dir = Path('cleaned_data/analysis_plots')
output_dir.mkdir(exist_ok=True)

# ============================================================================
# 1. Performance Over Time Comparison
# ============================================================================

print("\n📊 Creating performance over time plots...")

fig, axes = plt.subplots(5, 2, figsize=(16, 20))
fig.suptitle('Model Performance Over Time: Baseline Ridge vs XGBoost', fontsize=16, fontweight='bold')

commodities = baseline_results['commodity'].unique()

for idx, commodity in enumerate(commodities):
    baseline_comm = baseline_results[baseline_results['commodity'] == commodity].copy()
    xgb_comm = xgb_results[xgb_results['commodity'] == commodity].copy()
    
    baseline_comm['test_week'] = pd.to_datetime(baseline_comm['test_week'])
    xgb_comm['test_week'] = pd.to_datetime(xgb_comm['test_week'])
    
    # R² over time
    ax = axes[idx, 0]
    ax.plot(baseline_comm['test_week'], baseline_comm['r2'], label='Ridge', marker='o', alpha=0.7)
    ax.plot(xgb_comm['test_week'], xgb_comm['r2'], label='XGBoost', marker='s', alpha=0.7)
    ax.set_title(f'{commodity} - R² Score Over Time')
    ax.set_xlabel('Test Week')
    ax.set_ylabel('R²')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)
    
    # Accuracy over time
    ax = axes[idx, 1]
    ax.plot(baseline_comm['test_week'], baseline_comm['accuracy'], label='Ridge', marker='o', alpha=0.7)
    ax.plot(xgb_comm['test_week'], xgb_comm['accuracy'], label='XGBoost', marker='s', alpha=0.7)
    ax.set_title(f'{commodity} - Direction Accuracy Over Time')
    ax.set_xlabel('Test Week')
    ax.set_ylabel('Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, label='Random')

plt.tight_layout()
plt.savefig(output_dir / 'performance_over_time.png', dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: {output_dir / 'performance_over_time.png'}")

# ============================================================================
# 2. Predicted vs Actual Returns
# ============================================================================

print("\n📊 Creating predicted vs actual plots...")

fig, axes = plt.subplots(5, 2, figsize=(16, 20))
fig.suptitle('Predicted vs Actual Returns: Baseline Ridge vs XGBoost', fontsize=16, fontweight='bold')

for idx, commodity in enumerate(commodities):
    baseline_comm = baseline_predictions[baseline_predictions['commodity'] == commodity]
    xgb_comm = xgb_predictions[xgb_predictions['commodity'] == commodity]
    
    # Baseline Ridge
    ax = axes[idx, 0]
    ax.scatter(baseline_comm['actual_return'], baseline_comm['predicted_return'], alpha=0.5, s=20)
    ax.plot([baseline_comm['actual_return'].min(), baseline_comm['actual_return'].max()],
            [baseline_comm['actual_return'].min(), baseline_comm['actual_return'].max()],
            'r--', linewidth=2, label='Perfect Prediction')
    ax.set_title(f'{commodity} - Ridge Regression')
    ax.set_xlabel('Actual Return')
    ax.set_ylabel('Predicted Return')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # XGBoost
    ax = axes[idx, 1]
    ax.scatter(xgb_comm['actual_return'], xgb_comm['predicted_return'], alpha=0.5, s=20, color='green')
    ax.plot([xgb_comm['actual_return'].min(), xgb_comm['actual_return'].max()],
            [xgb_comm['actual_return'].min(), xgb_comm['actual_return'].max()],
            'r--', linewidth=2, label='Perfect Prediction')
    ax.set_title(f'{commodity} - XGBoost')
    ax.set_xlabel('Actual Return')
    ax.set_ylabel('Predicted Return')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'predicted_vs_actual.png', dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: {output_dir / 'predicted_vs_actual.png'}")

# ============================================================================
# 3. Time Series of Returns
# ============================================================================

print("\n📊 Creating time series plots...")

fig, axes = plt.subplots(5, 1, figsize=(16, 20))
fig.suptitle('Return Predictions Time Series: Baseline Ridge vs XGBoost', fontsize=16, fontweight='bold')

for idx, commodity in enumerate(commodities):
    baseline_comm = baseline_predictions[baseline_predictions['commodity'] == commodity].sort_values('date')
    xgb_comm = xgb_predictions[xgb_predictions['commodity'] == commodity].sort_values('date')
    
    ax = axes[idx]
    ax.plot(baseline_comm['date'], baseline_comm['actual_return'], 
            label='Actual', color='black', linewidth=2, alpha=0.7)
    ax.plot(baseline_comm['date'], baseline_comm['predicted_return'], 
            label='Ridge Predicted', color='blue', linewidth=1.5, alpha=0.7)
    ax.plot(xgb_comm['date'], xgb_comm['predicted_return'], 
            label='XGBoost Predicted', color='green', linewidth=1.5, alpha=0.7)
    ax.set_title(f'{commodity} - Return Predictions Over Time')
    ax.set_xlabel('Date')
    ax.set_ylabel('Return')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)

plt.tight_layout()
plt.savefig(output_dir / 'time_series.png', dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: {output_dir / 'time_series.png'}")

# ============================================================================
# 4. Confusion Matrices
# ============================================================================

print("\n📊 Creating confusion matrices...")

from sklearn.metrics import confusion_matrix

fig, axes = plt.subplots(5, 2, figsize=(12, 22))
fig.suptitle('Direction Prediction Confusion Matrices', fontsize=16, fontweight='bold')

for idx, commodity in enumerate(commodities):
    baseline_comm = baseline_predictions[baseline_predictions['commodity'] == commodity]
    xgb_comm = xgb_predictions[xgb_predictions['commodity'] == commodity]
    
    # Baseline Ridge
    ax = axes[idx, 0]
    cm_baseline = confusion_matrix(baseline_comm['actual_direction'], baseline_comm['predicted_direction'])
    sns.heatmap(cm_baseline, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
    ax.set_title(f'{commodity} - Ridge')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_xticklabels(['Down', 'Up'])
    ax.set_yticklabels(['Down', 'Up'])
    
    # XGBoost
    ax = axes[idx, 1]
    cm_xgb = confusion_matrix(xgb_comm['actual_direction'], xgb_comm['predicted_direction'])
    sns.heatmap(cm_xgb, annot=True, fmt='d', cmap='Greens', ax=ax, cbar=False)
    ax.set_title(f'{commodity} - XGBoost')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_xticklabels(['Down', 'Up'])
    ax.set_yticklabels(['Down', 'Up'])

plt.tight_layout()
plt.savefig(output_dir / 'confusion_matrices.png', dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: {output_dir / 'confusion_matrices.png'}")

# ============================================================================
# 5. Feature Importance Comparison
# ============================================================================

print("\n📊 Creating feature importance plots...")

# Load feature importance from XGBoost artifacts
import json

with open(xgb_dir / 'feature_importance.json', 'r') as f:
    xgb_importance = json.load(f)

# Load coefficients from Ridge artifacts
import joblib

fig, axes = plt.subplots(5, 2, figsize=(16, 22))
fig.suptitle('Feature Importance: Ridge Coefficients vs XGBoost Gain', fontsize=16, fontweight='bold')

for idx, commodity in enumerate(commodities):
    # Ridge coefficients (from last fold)
    artifacts_files = sorted(baseline_dir.glob(f"{commodity}_fold*_artifacts.joblib"))
    if artifacts_files:
        last_artifact = joblib.load(artifacts_files[-1])
        feature_names = last_artifact['feature_names']
        coefficients = last_artifact['reg_coefficients']
        
        coef_df = pd.DataFrame({
            'feature': feature_names,
            'coefficient': np.abs(coefficients)
        }).sort_values('coefficient', ascending=False).head(15)
        
        ax = axes[idx, 0]
        ax.barh(range(len(coef_df)), coef_df['coefficient'], color='steelblue')
        ax.set_yticks(range(len(coef_df)))
        ax.set_yticklabels(coef_df['feature'], fontsize=8)
        ax.set_xlabel('Absolute Coefficient')
        ax.set_title(f'{commodity} - Ridge (Top 15 Features)')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis='x')
    
    # XGBoost gain
    commodity_importance = [item for item in xgb_importance if item['commodity'] == commodity and item['type'] == 'gain']
    if commodity_importance:
        importance_records = commodity_importance[0]['importance'][:15]
        
        ax = axes[idx, 1]
        features = [r['feature'] for r in importance_records]
        gains = [r['gain'] for r in importance_records]
        ax.barh(range(len(features)), gains, color='forestgreen')
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels(features, fontsize=8)
        ax.set_xlabel('Gain')
        ax.set_title(f'{commodity} - XGBoost (Top 15 Features)')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig(output_dir / 'feature_importance.png', dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: {output_dir / 'feature_importance.png'}")

# ============================================================================
# 6. Distribution of Errors
# ============================================================================

print("\n📊 Creating error distribution plots...")

fig, axes = plt.subplots(5, 2, figsize=(16, 20))
fig.suptitle('Distribution of Prediction Errors', fontsize=16, fontweight='bold')

for idx, commodity in enumerate(commodities):
    baseline_comm = baseline_predictions[baseline_predictions['commodity'] == commodity]
    xgb_comm = xgb_predictions[xgb_predictions['commodity'] == commodity]
    
    baseline_errors = baseline_comm['predicted_return'] - baseline_comm['actual_return']
    xgb_errors = xgb_comm['predicted_return'] - xgb_comm['actual_return']
    
    # Histogram
    ax = axes[idx, 0]
    ax.hist(baseline_errors, bins=30, alpha=0.6, label='Ridge', color='blue')
    ax.hist(xgb_errors, bins=30, alpha=0.6, label='XGBoost', color='green')
    ax.set_title(f'{commodity} - Error Distribution')
    ax.set_xlabel('Prediction Error')
    ax.set_ylabel('Frequency')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2)
    
    # Box plot
    ax = axes[idx, 1]
    ax.boxplot([baseline_errors, xgb_errors], labels=['Ridge', 'XGBoost'])
    ax.set_title(f'{commodity} - Error Box Plot')
    ax.set_ylabel('Prediction Error')
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='red', linestyle='--', linewidth=2)

plt.tight_layout()
plt.savefig(output_dir / 'error_distributions.png', dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: {output_dir / 'error_distributions.png'}")

# ============================================================================
# Summary Statistics
# ============================================================================

print("\n" + "="*80)
print("PERFORMANCE SUMMARY")
print("="*80)

print("\n📊 BASELINE RIDGE MODEL")
print("-" * 80)
print("\nOverall Performance:")
print(f"  MAE:      {baseline_results['mae'].mean():.4f} ± {baseline_results['mae'].std():.4f}")
print(f"  RMSE:     {baseline_results['rmse'].mean():.4f} ± {baseline_results['rmse'].std():.4f}")
print(f"  R²:       {baseline_results['r2'].mean():.4f} ± {baseline_results['r2'].std():.4f}")
print(f"  Accuracy: {baseline_results['accuracy'].mean():.4f} ± {baseline_results['accuracy'].std():.4f}")
print(f"  F1 Score: {baseline_results['f1'].mean():.4f} ± {baseline_results['f1'].std():.4f}")

print("\nBy Commodity:")
for commodity in commodities:
    comm_results = baseline_results[baseline_results['commodity'] == commodity]
    print(f"\n{commodity}:")
    print(f"  MAE:      {comm_results['mae'].mean():.4f}")
    print(f"  R²:       {comm_results['r2'].mean():.4f}")
    print(f"  Accuracy: {comm_results['accuracy'].mean():.4f}")

print("\n" + "="*80)
print("📊 XGBOOST MODEL")
print("-" * 80)
print("\nOverall Performance:")
print(f"  MAE:        {xgb_results['mae'].mean():.4f} ± {xgb_results['mae'].std():.4f}")
print(f"  RMSE:       {xgb_results['rmse'].mean():.4f} ± {xgb_results['rmse'].std():.4f}")
print(f"  R²:         {xgb_results['r2'].mean():.4f} ± {xgb_results['r2'].std():.4f}")
print(f"  Accuracy:   {xgb_results['accuracy'].mean():.4f} ± {xgb_results['accuracy'].std():.4f}")
print(f"  F1 Score:   {xgb_results['f1'].mean():.4f} ± {xgb_results['f1'].std():.4f}")
print(f"  ROC-AUC:    {xgb_results['roc_auc'].mean():.4f} ± {xgb_results['roc_auc'].std():.4f}")
print(f"  Brier:      {xgb_results['brier_score'].mean():.4f} ± {xgb_results['brier_score'].std():.4f}")

print("\nBy Commodity:")
for commodity in commodities:
    comm_results = xgb_results[xgb_results['commodity'] == commodity]
    print(f"\n{commodity}:")
    print(f"  MAE:      {comm_results['mae'].mean():.4f}")
    print(f"  R²:       {comm_results['r2'].mean():.4f}")
    print(f"  Accuracy: {comm_results['accuracy'].mean():.4f}")
    print(f"  ROC-AUC:  {comm_results['roc_auc'].mean():.4f}")

print("\n" + "="*80)
print("📊 MODEL COMPARISON")
print("-" * 80)

comparison = pd.DataFrame({
    'Metric': ['MAE', 'RMSE', 'R²', 'Accuracy', 'F1'],
    'Ridge': [
        baseline_results['mae'].mean(),
        baseline_results['rmse'].mean(),
        baseline_results['r2'].mean(),
        baseline_results['accuracy'].mean(),
        baseline_results['f1'].mean()
    ],
    'XGBoost': [
        xgb_results['mae'].mean(),
        xgb_results['rmse'].mean(),
        xgb_results['r2'].mean(),
        xgb_results['accuracy'].mean(),
        xgb_results['f1'].mean()
    ]
})

comparison['Improvement'] = ((comparison['XGBoost'] - comparison['Ridge']) / comparison['Ridge'] * 100).round(2)
comparison['Winner'] = comparison.apply(
    lambda row: 'XGBoost' if row['Improvement'] > 0 else 'Ridge' if row['Improvement'] < 0 else 'Tie', axis=1
)

print(f"\n{comparison.to_string(index=False)}")

print("\n" + "="*80)
print("✅ ANALYSIS COMPLETE!")
print(f"📁 All plots saved to: {output_dir}")
print("="*80 + "\n")
