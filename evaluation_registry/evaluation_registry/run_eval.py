"""
Run Evaluation - Standalone Version
Just run: python run_eval.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, f1_score, roc_auc_score, brier_score_loss
)
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# CONFIGURATION
# ============================================================================

PREDICTIONS_FILE = 'cleaned_data/xgboost_artifacts/xgb_predictions.csv'
MODEL_NAME = 'xgb'
VERSION = 'v1'
OUTPUT_DIR = Path('reports')


# ============================================================================
# Evaluation Functions
# ============================================================================

def evaluate_commodity(df, commodity):
    """Evaluate predictions for one commodity."""
    
    y_true_ret = df['actual_return'].values
    y_pred_ret = df['predicted_return'].values
    y_true_dir = df['actual_direction'].values
    y_pred_dir = df['predicted_direction'].values
    
    # Magnitude metrics
    mae = mean_absolute_error(y_true_ret, y_pred_ret)
    rmse = np.sqrt(mean_squared_error(y_true_ret, y_pred_ret))
    r2 = r2_score(y_true_ret, y_pred_ret)
    
    # Direction metrics
    accuracy = accuracy_score(y_true_dir, y_pred_dir)
    f1 = f1_score(y_true_dir, y_pred_dir, zero_division=0)
    
    # Probability-based metrics
    roc_auc = None
    brier = None
    
    proba_col = None
    for col in ['predicted_proba', 'direction_probability']:
        if col in df.columns:
            proba_col = col
            break
    
    if proba_col and len(np.unique(y_true_dir)) > 1:
        y_proba = df[proba_col].values
        try:
            roc_auc = roc_auc_score(y_true_dir, y_proba)
            brier = brier_score_loss(y_true_dir, y_proba)
        except:
            pass
    
    return {
        'commodity': commodity,
        'n_samples': len(df),
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'accuracy': accuracy,
        'f1': f1,
        'roc_auc': roc_auc,
        'brier_score': brier
    }


def main():
    print("="*70)
    print("UNIFIED EVALUATION PIPELINE")
    print("="*70)
    
    # Check file exists
    if not Path(PREDICTIONS_FILE).exists():
        print(f"\n❌ File not found: {PREDICTIONS_FILE}")
        
        # Look for alternatives
        print("\nSearching for predictions file...")
        for path in Path('cleaned_data').rglob('*predictions*.csv'):
            print(f"  Found: {path}")
        return
    
    # Load predictions
    print(f"\n📂 Loading: {PREDICTIONS_FILE}")
    df = pd.read_csv(PREDICTIONS_FILE)
    print(f"  Rows: {len(df)}")
    print(f"  Columns: {df.columns.tolist()}")
    print(f"  Commodities: {df['commodity'].unique().tolist()}")
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Evaluate each commodity
    print("\n" + "="*70)
    print("📊 EVALUATION RESULTS")
    print("="*70)
    
    results = []
    
    for commodity in df['commodity'].unique():
        comm_df = df[df['commodity'] == commodity]
        result = evaluate_commodity(comm_df, commodity)
        results.append(result)
        
        print(f"\n{commodity}:")
        print(f"  Samples:  {result['n_samples']}")
        print(f"  MAE:      {result['mae']:.4f}")
        print(f"  RMSE:     {result['rmse']:.4f}")
        print(f"  R²:       {result['r2']:.4f}")
        print(f"  Accuracy: {result['accuracy']:.4f}")
        print(f"  F1:       {result['f1']:.4f}")
        if result['roc_auc']:
            print(f"  ROC-AUC:  {result['roc_auc']:.4f}")
        if result['brier_score']:
            print(f"  Brier:    {result['brier_score']:.4f}")
    
    # Save results summary
    results_df = pd.DataFrame(results)
    results_path = OUTPUT_DIR / 'evaluation_results.csv'
    results_df.to_csv(results_path, index=False)
    print(f"\n💾 Results saved to: {results_path}")
    
    # Export OOS predictions in standard format
    print("\n📄 Exporting OOS predictions...")
    
    oos_df = df.copy()
    
    # Rename columns to standard format
    rename_map = {
        'actual_return': 'y_true',
        'predicted_return': 'y_pred_ret',
        'predicted_proba': 'y_pred_dir_prob',
        'direction_probability': 'y_pred_dir_prob',
        'week': 'week_start',
        'date': 'week_start'
    }
    
    for old, new in rename_map.items():
        if old in oos_df.columns:
            oos_df = oos_df.rename(columns={old: new})
    
    oos_df['model_name'] = MODEL_NAME
    
    # Select output columns
    out_cols = ['commodity', 'week_start', 'y_true', 'y_pred_ret', 'y_pred_dir_prob', 'model_name']
    out_cols = [c for c in out_cols if c in oos_df.columns]
    
    oos_path = OUTPUT_DIR / 'oos_predictions.csv'
    oos_df[out_cols].to_csv(oos_path, index=False)
    print(f"  Saved to: {oos_path}")
    
    # Overall summary
    print("\n" + "="*70)
    print("📊 OVERALL SUMMARY")
    print("="*70)
    print(f"\n  Avg MAE:      {results_df['mae'].mean():.4f}")
    print(f"  Avg RMSE:     {results_df['rmse'].mean():.4f}")
    print(f"  Avg R²:       {results_df['r2'].mean():.4f}")
    print(f"  Avg Accuracy: {results_df['accuracy'].mean():.4f}")
    print(f"  Avg F1:       {results_df['f1'].mean():.4f}")
    
    print("\n" + "="*70)
    print("✅ EVALUATION COMPLETE!")
    print("="*70)
    print(f"\nOutputs:")
    print(f"  - {results_path}")
    print(f"  - {oos_path}")


if __name__ == '__main__':
    main()
