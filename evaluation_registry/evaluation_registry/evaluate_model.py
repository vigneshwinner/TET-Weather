"""
Unified Evaluation Pipeline
Common evaluation functions for consistent model comparison.

Issue #17: Unified Evaluation and Model Registry
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, f1_score, roc_auc_score, average_precision_score,
    brier_score_loss, log_loss, precision_score, recall_score,
    confusion_matrix
)
import warnings
warnings.filterwarnings('ignore')


@dataclass
class MagnitudeMetrics:
    """Regression metrics for magnitude prediction."""
    mae: float
    rmse: float
    r2: float
    mape: Optional[float] = None
    directional_accuracy: Optional[float] = None  # % where sign matches
    
    def to_dict(self) -> Dict:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass 
class DirectionMetrics:
    """Classification metrics for direction prediction."""
    accuracy: float
    f1: float
    precision: float
    recall: float
    roc_auc: Optional[float] = None
    pr_auc: Optional[float] = None
    brier_score: Optional[float] = None
    log_loss_val: Optional[float] = None
    hit_ratio: Optional[float] = None  # Same as accuracy, for clarity
    
    def to_dict(self) -> Dict:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class EvaluationResult:
    """Complete evaluation result for a model."""
    commodity: str
    model_name: str
    fold: Optional[int]
    test_week: Optional[str]
    magnitude_metrics: MagnitudeMetrics
    direction_metrics: DirectionMetrics
    n_samples: int
    
    def to_dict(self) -> Dict:
        return {
            'commodity': self.commodity,
            'model_name': self.model_name,
            'fold': self.fold,
            'test_week': self.test_week,
            'n_samples': self.n_samples,
            **{f'mag_{k}': v for k, v in self.magnitude_metrics.to_dict().items()},
            **{f'dir_{k}': v for k, v in self.direction_metrics.to_dict().items()},
        }


def evaluate_magnitude(
    y_true: np.ndarray, 
    y_pred: np.ndarray,
    y_true_direction: Optional[np.ndarray] = None
) -> MagnitudeMetrics:
    """
    Evaluate magnitude (regression) predictions.
    
    Args:
        y_true: True return values
        y_pred: Predicted return values
        y_true_direction: Optional true direction labels for directional accuracy
    
    Returns:
        MagnitudeMetrics with MAE, RMSE, R2, and optional MAPE/directional accuracy
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    # MAPE (handle zeros)
    non_zero_mask = y_true != 0
    if non_zero_mask.sum() > 0:
        mape = np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100
    else:
        mape = None
    
    # Directional accuracy from magnitude predictions
    if y_true_direction is not None:
        pred_direction = (y_pred > 0).astype(int)
        directional_accuracy = accuracy_score(y_true_direction, pred_direction)
    else:
        # Infer direction from returns
        true_direction = (y_true > 0).astype(int)
        pred_direction = (y_pred > 0).astype(int)
        directional_accuracy = accuracy_score(true_direction, pred_direction)
    
    return MagnitudeMetrics(
        mae=mae,
        rmse=rmse,
        r2=r2,
        mape=mape,
        directional_accuracy=directional_accuracy
    )


def evaluate_direction(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: Optional[np.ndarray] = None
) -> DirectionMetrics:
    """
    Evaluate direction (classification) predictions.
    
    Args:
        y_true: True direction labels (0/1)
        y_pred: Predicted direction labels (0/1)
        y_pred_proba: Optional predicted probabilities for class 1
    
    Returns:
        DirectionMetrics with accuracy, F1, ROC-AUC, PR-AUC, Brier score, etc.
    """
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    
    # Probability-based metrics
    roc_auc = None
    pr_auc = None
    brier = None
    logloss = None
    
    if y_pred_proba is not None:
        # Check if we have both classes in y_true
        if len(np.unique(y_true)) > 1:
            try:
                roc_auc = roc_auc_score(y_true, y_pred_proba)
            except Exception:
                pass
            
            try:
                pr_auc = average_precision_score(y_true, y_pred_proba)
            except Exception:
                pass
        
        try:
            brier = brier_score_loss(y_true, y_pred_proba)
        except Exception:
            pass
        
        try:
            # Clip probabilities to avoid log(0)
            proba_clipped = np.clip(y_pred_proba, 1e-10, 1 - 1e-10)
            logloss = log_loss(y_true, proba_clipped)
        except Exception:
            pass
    
    return DirectionMetrics(
        accuracy=accuracy,
        f1=f1,
        precision=precision,
        recall=recall,
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        brier_score=brier,
        log_loss_val=logloss,
        hit_ratio=accuracy
    )


def evaluate_model(
    y_true_return: np.ndarray,
    y_pred_return: np.ndarray,
    y_true_direction: np.ndarray,
    y_pred_direction: np.ndarray,
    y_pred_direction_proba: Optional[np.ndarray] = None,
    commodity: str = "unknown",
    model_name: str = "unknown",
    fold: Optional[int] = None,
    test_week: Optional[str] = None
) -> EvaluationResult:
    """
    Comprehensive model evaluation for both magnitude and direction.
    
    Args:
        y_true_return: True return values
        y_pred_return: Predicted return values
        y_true_direction: True direction labels (0/1)
        y_pred_direction: Predicted direction labels (0/1)
        y_pred_direction_proba: Predicted probabilities for positive class
        commodity: Commodity name
        model_name: Model identifier
        fold: Fold number (for CV)
        test_week: Test week identifier
    
    Returns:
        EvaluationResult containing all metrics
    """
    magnitude_metrics = evaluate_magnitude(
        y_true_return, 
        y_pred_return, 
        y_true_direction
    )
    
    direction_metrics = evaluate_direction(
        y_true_direction,
        y_pred_direction,
        y_pred_direction_proba
    )
    
    return EvaluationResult(
        commodity=commodity,
        model_name=model_name,
        fold=fold,
        test_week=test_week,
        magnitude_metrics=magnitude_metrics,
        direction_metrics=direction_metrics,
        n_samples=len(y_true_return)
    )


def aggregate_fold_results(results: List[EvaluationResult]) -> Dict[str, Any]:
    """
    Aggregate results across multiple folds.
    
    Args:
        results: List of EvaluationResult from each fold
    
    Returns:
        Dictionary with mean and std for each metric
    """
    if not results:
        return {}
    
    # Collect all metrics into lists
    metrics_dict = {}
    for result in results:
        result_dict = result.to_dict()
        for key, value in result_dict.items():
            if isinstance(value, (int, float)) and value is not None:
                if key not in metrics_dict:
                    metrics_dict[key] = []
                metrics_dict[key].append(value)
    
    # Calculate mean and std
    aggregated = {
        'commodity': results[0].commodity,
        'model_name': results[0].model_name,
        'n_folds': len(results),
        'total_samples': sum(r.n_samples for r in results)
    }
    
    for key, values in metrics_dict.items():
        if key not in ['fold', 'n_samples']:
            values_arr = np.array([v for v in values if v is not None and not np.isnan(v)])
            if len(values_arr) > 0:
                aggregated[f'{key}_mean'] = float(np.mean(values_arr))
                aggregated[f'{key}_std'] = float(np.std(values_arr))
    
    return aggregated


def compare_models(
    model_results: Dict[str, List[EvaluationResult]],
    primary_metric: str = 'dir_accuracy'
) -> pd.DataFrame:
    """
    Compare multiple models across all commodities.
    
    Args:
        model_results: Dict mapping model_name to list of EvaluationResults
        primary_metric: Metric to sort by
    
    Returns:
        DataFrame comparing models
    """
    comparison_rows = []
    
    for model_name, results in model_results.items():
        # Group by commodity
        by_commodity = {}
        for r in results:
            if r.commodity not in by_commodity:
                by_commodity[r.commodity] = []
            by_commodity[r.commodity].append(r)
        
        for commodity, commodity_results in by_commodity.items():
            agg = aggregate_fold_results(commodity_results)
            agg['model_name'] = model_name
            comparison_rows.append(agg)
    
    df = pd.DataFrame(comparison_rows)
    
    # Sort by primary metric
    sort_col = f'{primary_metric}_mean'
    if sort_col in df.columns:
        df = df.sort_values(sort_col, ascending=False)
    
    return df


def generate_evaluation_summary(
    results: List[EvaluationResult],
    output_format: str = 'dict'
) -> Any:
    """
    Generate a comprehensive evaluation summary.
    
    Args:
        results: List of evaluation results
        output_format: 'dict', 'dataframe', or 'markdown'
    
    Returns:
        Summary in requested format
    """
    # Aggregate overall
    overall = aggregate_fold_results(results)
    
    # Group by commodity
    by_commodity = {}
    for r in results:
        if r.commodity not in by_commodity:
            by_commodity[r.commodity] = []
        by_commodity[r.commodity].append(r)
    
    commodity_summaries = {
        commodity: aggregate_fold_results(commodity_results)
        for commodity, commodity_results in by_commodity.items()
    }
    
    summary = {
        'overall': overall,
        'by_commodity': commodity_summaries
    }
    
    if output_format == 'dataframe':
        rows = [{'commodity': 'OVERALL', **overall}]
        for commodity, comm_summary in commodity_summaries.items():
            rows.append({'commodity': commodity, **comm_summary})
        return pd.DataFrame(rows)
    
    elif output_format == 'markdown':
        md = "# Model Evaluation Summary\n\n"
        md += f"**Model:** {overall.get('model_name', 'Unknown')}\n"
        md += f"**Total Folds:** {overall.get('n_folds', 0)}\n"
        md += f"**Total Samples:** {overall.get('total_samples', 0)}\n\n"
        
        md += "## Overall Metrics\n\n"
        md += "| Metric | Mean | Std |\n"
        md += "|--------|------|-----|\n"
        for key in ['mag_mae', 'mag_rmse', 'mag_r2', 'dir_accuracy', 'dir_f1', 'dir_roc_auc', 'dir_brier_score']:
            mean_key = f'{key}_mean'
            std_key = f'{key}_std'
            if mean_key in overall:
                md += f"| {key} | {overall[mean_key]:.4f} | {overall.get(std_key, 0):.4f} |\n"
        
        md += "\n## By Commodity\n\n"
        for commodity, comm_summary in commodity_summaries.items():
            md += f"### {commodity}\n\n"
            md += f"- MAE: {comm_summary.get('mag_mae_mean', 0):.4f}\n"
            md += f"- R²: {comm_summary.get('mag_r2_mean', 0):.4f}\n"
            md += f"- Accuracy: {comm_summary.get('dir_accuracy_mean', 0):.4f}\n"
            md += f"- ROC-AUC: {comm_summary.get('dir_roc_auc_mean', 0):.4f}\n"
            md += f"- Brier: {comm_summary.get('dir_brier_score_mean', 0):.4f}\n\n"
        
        return md
    
    return summary


# ============================================================================
# Convenience function matching Issue #17 spec
# ============================================================================

def evaluate_model_simple(
    preds: pd.DataFrame,
    truth: pd.DataFrame,
    commodity: str = "unknown",
    model_name: str = "unknown"
) -> Dict[str, Any]:
    """
    Simple evaluation interface matching Issue #17 specification.
    
    Args:
        preds: DataFrame with columns [predicted_return, predicted_direction, predicted_proba]
        truth: DataFrame with columns [actual_return, actual_direction]
        commodity: Commodity name
        model_name: Model identifier
    
    Returns:
        Dictionary with all metrics:
        - Magnitude: MAE, RMSE, R2
        - Direction: accuracy, F1, ROC-AUC, PR-AUC, Brier, hit_ratio
    """
    result = evaluate_model(
        y_true_return=truth['actual_return'].values,
        y_pred_return=preds['predicted_return'].values,
        y_true_direction=truth['actual_direction'].values,
        y_pred_direction=preds['predicted_direction'].values,
        y_pred_direction_proba=preds.get('predicted_proba', preds.get('direction_probability')).values if 'predicted_proba' in preds.columns or 'direction_probability' in preds.columns else None,
        commodity=commodity,
        model_name=model_name
    )
    
    return result.to_dict()


# ============================================================================
# Demo
# ============================================================================

if __name__ == '__main__':
    print("="*80)
    print("EVALUATION MODULE DEMO")
    print("="*80)
    
    # Generate sample data
    np.random.seed(42)
    n = 100
    
    y_true_return = np.random.randn(n) * 0.05
    y_pred_return = y_true_return + np.random.randn(n) * 0.02
    
    y_true_direction = (y_true_return > 0).astype(int)
    y_pred_proba = 1 / (1 + np.exp(-y_pred_return * 20))  # Sigmoid
    y_pred_direction = (y_pred_proba > 0.5).astype(int)
    
    # Evaluate
    result = evaluate_model(
        y_true_return=y_true_return,
        y_pred_return=y_pred_return,
        y_true_direction=y_true_direction,
        y_pred_direction=y_pred_direction,
        y_pred_direction_proba=y_pred_proba,
        commodity="Henry_Hub",
        model_name="XGBoost_v1",
        fold=1,
        test_week="2024-01-01"
    )
    
    print("\n📊 Evaluation Result:")
    print(f"\nMagnitude Metrics:")
    print(f"  MAE:  {result.magnitude_metrics.mae:.4f}")
    print(f"  RMSE: {result.magnitude_metrics.rmse:.4f}")
    print(f"  R²:   {result.magnitude_metrics.r2:.4f}")
    print(f"  Directional Accuracy: {result.magnitude_metrics.directional_accuracy:.4f}")
    
    print(f"\nDirection Metrics:")
    print(f"  Accuracy:    {result.direction_metrics.accuracy:.4f}")
    print(f"  F1:          {result.direction_metrics.f1:.4f}")
    print(f"  ROC-AUC:     {result.direction_metrics.roc_auc:.4f}")
    print(f"  PR-AUC:      {result.direction_metrics.pr_auc:.4f}")
    print(f"  Brier Score: {result.direction_metrics.brier_score:.4f}")
    
    print("\n✅ Demo complete!")
