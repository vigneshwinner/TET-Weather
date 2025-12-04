"""
Run Evaluation - Simplified Version
Just run: python run_evaluation_simple.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import sys
import os

# Add evaluation_registry to path
sys.path.insert(0, str(Path(__file__).parent / 'evaluation_registry'))

from evaluation_registry.evaluate_model import (
    evaluate_model, 
    aggregate_fold_results,
    generate_evaluation_summary
)
from evaluation_registry.leakage_checks import LeakageChecker
from evaluation_registry.error_analysis import generate_error_analysis_report
from evaluation_registry.model_registry import ModelRegistry


# ============================================================================
# CONFIGURATION - Edit these paths if needed
# ============================================================================

PREDICTIONS_FILE = 'cleaned_data/xgboost_artifacts/predictions.csv'
MODEL_NAME = 'xgb'
VERSION = 'v1'
OUTPUT_DIR = Path('reports')
REGISTER_MODEL = True
SET_CHAMPION = True


# ============================================================================
# Main Script
# ============================================================================

def load_predictions(filepath):
    """Load and standardize predictions file."""
    print(f"📂 Loading predictions from: {filepath}")
    
    df = pd.read_csv(filepath)
    print(f"  Columns found: {df.columns.tolist()}")
    
    # Standardize column names (handle various naming conventions)
    column_mapping = {
        'actual_return': 'actual_return',
        'predicted_return': 'predicted_return',
        'actual_direction': 'actual_direction',
        'predicted_direction': 'predicted_direction',
        'predicted_proba': 'predicted_proba',
        'direction_probability': 'predicted_proba',
        'date': 'date',
        'week': 'week',
        'fold': 'fold'
    }
    
    rename_dict = {k: v for k, v in column_mapping.items() if k in df.columns}
    df = df.rename(columns=rename_dict)
    
    # Convert date columns
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    if 'week' in df.columns:
        df['week'] = pd.to_datetime(df['week'])
    
    print(f"  Loaded {len(df)} predictions")
    print(f"  Commodities: {df['commodity'].unique().tolist()}")
    
    return df


def run_evaluation(predictions_df, model_name):
    """Run evaluation for all commodities and folds."""
    
    print("\n" + "="*60)
    print("📊 RUNNING MODEL EVALUATION")
    print("="*60)
    
    all_results = []
    commodities = predictions_df['commodity'].unique()
    
    for commodity in commodities:
        comm_df = predictions_df[predictions_df['commodity'] == commodity]
        
        print(f"\n{commodity}:")
        
        # Check if we have folds
        if 'fold' in comm_df.columns:
            folds = comm_df['fold'].unique()
        else:
            folds = [None]
        
        for fold in folds:
            if fold is not None:
                fold_df = comm_df[comm_df['fold'] == fold]
            else:
                fold_df = comm_df
            
            if len(fold_df) == 0:
                continue
            
            # Get test week
            test_week = None
            if 'week' in fold_df.columns:
                test_week = str(fold_df['week'].iloc[0].date()) if len(fold_df) > 0 else None
            elif 'date' in fold_df.columns:
                test_week = str(fold_df['date'].iloc[0].date()) if len(fold_df) > 0 else None
            
            # Get probability column
            proba_col = None
            for col in ['predicted_proba', 'direction_probability', 'predicted_proba']:
                if col in fold_df.columns:
                    proba_col = col
                    break
            
            # Evaluate
            result = evaluate_model(
                y_true_return=fold_df['actual_return'].values,
                y_pred_return=fold_df['predicted_return'].values,
                y_true_direction=fold_df['actual_direction'].values,
                y_pred_direction=fold_df['predicted_direction'].values,
                y_pred_direction_proba=fold_df[proba_col].values if proba_col else None,
                commodity=commodity,
                model_name=model_name,
                fold=fold,
                test_week=test_week
            )
            
            all_results.append(result)
        
        # Aggregate for this commodity
        comm_results = [r for r in all_results if r.commodity == commodity]
        if comm_results:
            agg = aggregate_fold_results(comm_results)
            
            print(f"  Folds: {len(comm_results)}")
            print(f"  MAE:      {agg.get('mag_mae_mean', 0):.4f} ± {agg.get('mag_mae_std', 0):.4f}")
            print(f"  R²:       {agg.get('mag_r2_mean', 0):.4f} ± {agg.get('mag_r2_std', 0):.4f}")
            print(f"  Accuracy: {agg.get('dir_accuracy_mean', 0):.4f} ± {agg.get('dir_accuracy_std', 0):.4f}")
            if 'dir_roc_auc_mean' in agg:
                print(f"  ROC-AUC:  {agg.get('dir_roc_auc_mean', 0):.4f} ± {agg.get('dir_roc_auc_std', 0):.4f}")
            if 'dir_brier_score_mean' in agg:
                print(f"  Brier:    {agg.get('dir_brier_score_mean', 0):.4f} ± {agg.get('dir_brier_score_std', 0):.4f}")
    
    return {
        'results': all_results,
        'by_commodity': {
            commodity: aggregate_fold_results([r for r in all_results if r.commodity == commodity])
            for commodity in commodities
        },
        'overall': aggregate_fold_results(all_results)
    }


def export_oos_predictions(predictions_df, model_name, output_path):
    """Export predictions in standard OOS format."""
    
    print(f"\n💾 Exporting standard predictions to: {output_path}")
    
    output_df = predictions_df.copy()
    
    # Rename to standard columns
    rename_map = {
        'actual_return': 'y_true',
        'predicted_return': 'y_pred_ret',
        'predicted_proba': 'y_pred_dir_prob',
        'direction_probability': 'y_pred_dir_prob',
        'week': 'week_start',
        'date': 'week_start'
    }
    
    for old, new in rename_map.items():
        if old in output_df.columns and new not in output_df.columns:
            output_df = output_df.rename(columns={old: new})
    
    output_df['model_name'] = model_name
    
    # Select columns that exist
    output_cols = ['commodity', 'week_start', 'y_true', 'y_pred_ret', 'y_pred_dir_prob', 'model_name']
    output_cols = [c for c in output_cols if c in output_df.columns]
    
    output_df[output_cols].to_csv(output_path, index=False)
    print(f"  ✓ Saved {len(output_df)} rows")


def register_models(evaluation_results, model_name, version, set_champion):
    """Register models in the registry."""
    
    print(f"\n" + "="*60)
    print("📝 REGISTERING MODELS")
    print("="*60)
    
    registry = ModelRegistry('models/registry')
    
    for commodity, agg_metrics in evaluation_results['by_commodity'].items():
        metrics = {
            'mae': agg_metrics.get('mag_mae_mean', 0),
            'rmse': agg_metrics.get('mag_rmse_mean', 0),
            'r2': agg_metrics.get('mag_r2_mean', 0),
            'accuracy': agg_metrics.get('dir_accuracy_mean', 0),
            'f1': agg_metrics.get('dir_f1_mean', 0),
            'roc_auc': agg_metrics.get('dir_roc_auc_mean', 0),
            'pr_auc': agg_metrics.get('dir_pr_auc_mean', 0),
            'brier_score': agg_metrics.get('dir_brier_score_mean', 0)
        }
        
        # Clean NaN values
        metrics = {k: v if not (isinstance(v, float) and np.isnan(v)) else 0.0 
                   for k, v in metrics.items()}
        
        artifact_paths = {}
        artifacts = list(Path('cleaned_data/xgboost_artifacts').glob(f'{commodity}_fold*_xgb.joblib'))
        if artifacts:
            artifact_paths['latest_fold'] = str(artifacts[-1])
        
        registry.register_model(
            model_name=model_name,
            version=version,
            commodity=commodity,
            target='both',
            training_data_start='2022-12-05',
            training_data_end=datetime.now().strftime('%Y-%m-%d'),
            n_training_samples=agg_metrics.get('total_samples', 0),
            feature_list=['see_training_script'],
            metrics=metrics,
            artifact_paths=artifact_paths
        )
        
        if set_champion:
            registry.set_champion(commodity, model_name, version)
    
    print("\n" + registry.get_summary())


def main():
    print("="*70)
    print("UNIFIED EVALUATION PIPELINE")
    print("Issue #17: Unified Evaluation and Model Registry")
    print("="*70)
    
    # Check if predictions file exists
    if not Path(PREDICTIONS_FILE).exists():
        print(f"\n❌ Error: Predictions file not found: {PREDICTIONS_FILE}")
        print("\nLooking for alternatives...")
        
        # Try to find predictions file
        possible_paths = [
            'cleaned_data/xgboost_artifacts/predictions.csv',
            'cleaned_data/xgboost_artifacts/xgb_predictions.csv',
            'cleaned_data/model_artifacts/predictions.csv',
            'reports/oos_predictions.csv'
        ]
        
        for path in possible_paths:
            if Path(path).exists():
                print(f"  Found: {path}")
        
        return
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Load predictions
    predictions_df = load_predictions(PREDICTIONS_FILE)
    
    # 2. Run evaluation
    evaluation_results = run_evaluation(predictions_df, MODEL_NAME)
    
    # 3. Save evaluation summary
    summary_md = generate_evaluation_summary(evaluation_results['results'], output_format='markdown')
    summary_path = OUTPUT_DIR / f'{MODEL_NAME}_{VERSION}_evaluation.md'
    with open(summary_path, 'w') as f:
        f.write(summary_md)
    print(f"\n📄 Evaluation summary saved to: {summary_path}")
    
    # 4. Export standard OOS predictions
    oos_path = OUTPUT_DIR / 'oos_predictions.csv'
    export_oos_predictions(predictions_df, MODEL_NAME, oos_path)
    
    # 5. Generate plots (optional - comment out if matplotlib issues)
    try:
        print("\n📈 Generating plots...")
        plots_dir = OUTPUT_DIR / 'plots'
        plots_dir.mkdir(exist_ok=True)
        
        for commodity in predictions_df['commodity'].unique():
            comm_df = predictions_df[predictions_df['commodity'] == commodity].copy()
            
            if 'date' not in comm_df.columns and 'week' in comm_df.columns:
                comm_df['date'] = comm_df['week']
            
            try:
                saved_plots = generate_error_analysis_report(
                    predictions_df=comm_df,
                    commodity=commodity,
                    model_name=MODEL_NAME,
                    output_dir=plots_dir / commodity
                )
                print(f"  {commodity}: {len(saved_plots)} plots")
            except Exception as e:
                print(f"  {commodity}: Plot generation failed - {e}")
        
    except Exception as e:
        print(f"  ⚠️ Plot generation skipped: {e}")
    
    # 6. Register models
    if REGISTER_MODEL:
        register_models(evaluation_results, MODEL_NAME, VERSION, SET_CHAMPION)
    
    # Done!
    print("\n" + "="*70)
    print("✅ EVALUATION PIPELINE COMPLETE!")
    print("="*70)
    print(f"\nOutputs:")
    print(f"  - Evaluation summary: {summary_path}")
    print(f"  - OOS predictions: {oos_path}")
    print(f"  - Plots: {OUTPUT_DIR}/plots/")
    if REGISTER_MODEL:
        print(f"  - Model registry: models/registry/")


if __name__ == '__main__':
    main()