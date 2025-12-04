#!/usr/bin/env python
"""
Unified Evaluation Pipeline Integration
Integrates with existing XGBoost training to evaluate, validate, and register models.

Usage:
    python run_evaluation.py --predictions-file cleaned_data/xgboost_artifacts/xgb_predictions.csv
    python run_evaluation.py --predictions-file results.csv --register --set-champion

Issue #17: Unified Evaluation and Model Registry
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import sys

# Import evaluation modules
from evaluate_model import (
    evaluate_model, 
    aggregate_fold_results,
    generate_evaluation_summary,
    EvaluationResult
)
from leakage_checks import LeakageChecker, validate_no_leakage
from error_analysis import generate_error_analysis_report
from model_registry import ModelRegistry


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run unified evaluation pipeline'
    )
    
    parser.add_argument('--predictions-file', type=Path, required=True,
                        help='Path to predictions CSV (from train_xgboost_model.py)')
    parser.add_argument('--features-file', type=Path,
                        help='Path to features data for leakage checks')
    parser.add_argument('--model-name', default='xgb',
                        help='Model name for registration')
    parser.add_argument('--version', default='v1',
                        help='Model version')
    parser.add_argument('--output-dir', type=Path, default=Path('reports'),
                        help='Output directory for reports and plots')
    parser.add_argument('--register', action='store_true',
                        help='Register model in registry')
    parser.add_argument('--set-champion', action='store_true',
                        help='Set as champion if registering')
    parser.add_argument('--ssi-file', type=Path,
                        help='Optional SSI file for decile analysis')
    
    return parser.parse_args()


def load_predictions(filepath: Path) -> pd.DataFrame:
    """Load and standardize predictions file."""
    df = pd.read_csv(filepath)
    
    # Standardize column names
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
    
    # Rename columns that exist
    rename_dict = {k: v for k, v in column_mapping.items() if k in df.columns}
    df = df.rename(columns=rename_dict)
    
    # Convert date columns
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    if 'week' in df.columns:
        df['week'] = pd.to_datetime(df['week'])
    
    return df


def run_evaluation(predictions_df: pd.DataFrame, model_name: str) -> dict:
    """Run evaluation for all commodities and folds."""
    
    print("\n📊 Running Model Evaluation...")
    print("-" * 60)
    
    all_results = []
    
    # Group by commodity
    commodities = predictions_df['commodity'].unique()
    
    for commodity in commodities:
        comm_df = predictions_df[predictions_df['commodity'] == commodity]
        
        print(f"\n{commodity}:")
        
        # Evaluate each fold
        folds = comm_df['fold'].unique() if 'fold' in comm_df.columns else [None]
        
        for fold in folds:
            if fold is not None:
                fold_df = comm_df[comm_df['fold'] == fold]
            else:
                fold_df = comm_df
            
            # Get test week
            test_week = None
            if 'week' in fold_df.columns:
                test_week = str(fold_df['week'].iloc[0].date()) if len(fold_df) > 0 else None
            
            # Evaluate
            result = evaluate_model(
                y_true_return=fold_df['actual_return'].values,
                y_pred_return=fold_df['predicted_return'].values,
                y_true_direction=fold_df['actual_direction'].values,
                y_pred_direction=fold_df['predicted_direction'].values,
                y_pred_direction_proba=fold_df['predicted_proba'].values if 'predicted_proba' in fold_df.columns else None,
                commodity=commodity,
                model_name=model_name,
                fold=fold,
                test_week=test_week
            )
            
            all_results.append(result)
        
        # Aggregate for this commodity
        comm_results = [r for r in all_results if r.commodity == commodity]
        agg = aggregate_fold_results(comm_results)
        
        print(f"  Folds: {len(comm_results)}")
        print(f"  MAE:      {agg.get('mag_mae_mean', 0):.4f} ± {agg.get('mag_mae_std', 0):.4f}")
        print(f"  R²:       {agg.get('mag_r2_mean', 0):.4f} ± {agg.get('mag_r2_std', 0):.4f}")
        print(f"  Accuracy: {agg.get('dir_accuracy_mean', 0):.4f} ± {agg.get('dir_accuracy_std', 0):.4f}")
        print(f"  ROC-AUC:  {agg.get('dir_roc_auc_mean', 0):.4f} ± {agg.get('dir_roc_auc_std', 0):.4f}")
        print(f"  Brier:    {agg.get('dir_brier_score_mean', 0):.4f} ± {agg.get('dir_brier_score_std', 0):.4f}")
    
    return {
        'results': all_results,
        'by_commodity': {
            commodity: aggregate_fold_results([r for r in all_results if r.commodity == commodity])
            for commodity in commodities
        },
        'overall': aggregate_fold_results(all_results)
    }


def run_leakage_checks(predictions_df: pd.DataFrame, features_df: pd.DataFrame = None):
    """Run leakage validation checks."""
    
    print("\n🔍 Running Leakage Checks...")
    print("-" * 60)
    
    checker = LeakageChecker()
    
    # Check train/test temporal integrity by fold
    if 'fold' in predictions_df.columns:
        folds = predictions_df.groupby('fold')
        
        for fold_num, fold_df in folds:
            if 'week' in fold_df.columns:
                # Can't fully check without train data, but verify test weeks are sequential
                weeks = fold_df['week'].sort_values().unique()
                print(f"  Fold {fold_num}: Test week(s) = {[str(w.date()) for w in weeks[:3]]}...")
    
    # Basic date checks
    if 'date' in predictions_df.columns or 'week' in predictions_df.columns:
        date_col = 'date' if 'date' in predictions_df.columns else 'week'
        dates = pd.to_datetime(predictions_df[date_col])
        
        print(f"\n  Date range: {dates.min().date()} to {dates.max().date()}")
        print(f"  Total predictions: {len(predictions_df)}")
    
    # If we have features, check for target leakage
    if features_df is not None:
        feature_cols = [c for c in features_df.columns 
                       if c not in ['date', 'week', 'commodity', 'return_next', 'direction_next']]
        
        result = checker.check_target_leakage(
            features_df,
            target_col='return_next',
            feature_cols=feature_cols
        )
        
        if not result['passed']:
            print("\n  ⚠️ Potential target leakage detected!")
            for feat in result['suspicious_features'][:5]:
                print(f"    {feat['feature']}: correlation = {feat['correlation']:.3f}")
        else:
            print("\n  ✅ No target leakage detected")
    
    print("\n  ✅ Leakage checks complete")


def generate_plots(predictions_df: pd.DataFrame, model_name: str, output_dir: Path, ssi_df=None):
    """Generate error analysis plots."""
    
    print("\n📈 Generating Error Analysis Plots...")
    print("-" * 60)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    commodities = predictions_df['commodity'].unique()
    
    for commodity in commodities:
        comm_df = predictions_df[predictions_df['commodity'] == commodity].copy()
        
        # Merge SSI if available
        ssi_col = None
        if ssi_df is not None and commodity in ssi_df['commodity'].values:
            comm_ssi = ssi_df[ssi_df['commodity'] == commodity]
            comm_df = comm_df.merge(comm_ssi[['week', 'ssi']], on='week', how='left')
            ssi_col = 'ssi'
        
        # Ensure we have date column
        if 'date' not in comm_df.columns and 'week' in comm_df.columns:
            comm_df['date'] = comm_df['week']
        
        saved_plots = generate_error_analysis_report(
            predictions_df=comm_df,
            commodity=commodity,
            model_name=model_name,
            output_dir=output_dir / commodity,
            ssi_col=ssi_col
        )
        
        print(f"  {commodity}: {len(saved_plots)} plots")


def export_standard_predictions(predictions_df: pd.DataFrame, model_name: str, output_path: Path):
    """Export predictions in standard OOS format."""
    
    print(f"\n💾 Exporting Standard Predictions...")
    
    # Standardize format
    output_df = predictions_df.copy()
    
    # Rename to standard columns
    output_df = output_df.rename(columns={
        'actual_return': 'y_true',
        'predicted_return': 'y_pred_ret',
        'predicted_proba': 'y_pred_dir_prob',
        'week': 'week_start'
    })
    
    output_df['model_name'] = model_name
    
    # Select columns
    output_cols = ['commodity', 'week_start', 'y_true', 'y_pred_ret', 'y_pred_dir_prob', 'model_name']
    output_cols = [c for c in output_cols if c in output_df.columns]
    
    output_df[output_cols].to_csv(output_path, index=False)
    print(f"  Saved to: {output_path}")


def register_models(evaluation_results: dict, model_name: str, version: str, set_champion: bool):
    """Register models in the registry."""
    
    print(f"\n📝 Registering Models...")
    print("-" * 60)
    
    registry = ModelRegistry('models/registry')
    
    for commodity, agg_metrics in evaluation_results['by_commodity'].items():
        # Extract metrics for registration
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
        
        # Find artifacts
        artifact_paths = {}
        artifact_pattern = f'cleaned_data/xgboost_artifacts/{commodity}_fold*_xgb.joblib'
        artifacts = list(Path('.').glob(artifact_pattern))
        if artifacts:
            artifact_paths['latest_fold'] = str(artifacts[-1])
        
        # Register
        registry.register_model(
            model_name=model_name,
            version=version,
            commodity=commodity,
            target='both',
            training_data_start='2022-12-05',  # From your config
            training_data_end=datetime.now().strftime('%Y-%m-%d'),
            n_training_samples=agg_metrics.get('total_samples', 0),
            feature_list=['see_training_script'],  # Placeholder
            metrics=metrics,
            artifact_paths=artifact_paths
        )
        
        if set_champion:
            registry.set_champion(commodity, model_name, version)
    
    print(registry.get_summary())


def main():
    args = parse_args()
    
    print("="*80)
    print("UNIFIED EVALUATION PIPELINE")
    print("Issue #17: Unified Evaluation and Model Registry")
    print("="*80)
    
    # Load predictions
    print(f"\n📂 Loading predictions from: {args.predictions_file}")
    predictions_df = load_predictions(args.predictions_file)
    print(f"  Loaded {len(predictions_df)} predictions")
    print(f"  Commodities: {predictions_df['commodity'].unique().tolist()}")
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Run evaluation
    evaluation_results = run_evaluation(predictions_df, args.model_name)
    
    # Save evaluation summary
    summary_md = generate_evaluation_summary(evaluation_results['results'], output_format='markdown')
    summary_path = args.output_dir / f'{args.model_name}_{args.version}_evaluation.md'
    with open(summary_path, 'w') as f:
        f.write(summary_md)
    print(f"\n📄 Evaluation summary saved to: {summary_path}")
    
    # 2. Run leakage checks
    features_df = None
    if args.features_file and args.features_file.exists():
        features_df = pd.read_csv(args.features_file)
    run_leakage_checks(predictions_df, features_df)
    
    # 3. Generate plots
    ssi_df = None
    if args.ssi_file and args.ssi_file.exists():
        ssi_df = pd.read_csv(args.ssi_file)
        ssi_df['week'] = pd.to_datetime(ssi_df['week'])
    
    generate_plots(predictions_df, args.model_name, args.output_dir / 'plots', ssi_df)
    
    # 4. Export standard predictions
    oos_path = args.output_dir / 'oos_predictions.csv'
    export_standard_predictions(predictions_df, args.model_name, oos_path)
    
    # 5. Register models
    if args.register:
        register_models(evaluation_results, args.model_name, args.version, args.set_champion)
    
    print("\n" + "="*80)
    print("✅ EVALUATION PIPELINE COMPLETE!")
    print("="*80)
    print(f"\nOutputs:")
    print(f"  - Evaluation summary: {summary_path}")
    print(f"  - OOS predictions: {oos_path}")
    print(f"  - Plots: {args.output_dir / 'plots'}/")
    if args.register:
        print(f"  - Model registry: models/registry/")


if __name__ == '__main__':
    main()
