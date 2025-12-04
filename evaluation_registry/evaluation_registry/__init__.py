"""
Unified Evaluation and Model Registry
Issue #17

Components:
- evaluate_model: Common evaluation functions for magnitude and direction
- leakage_checks: Validation for temporal leakage and target leakage
- error_analysis: Diagnostic plots (prediction vs actual, residuals by SSI, lift charts)
- model_registry: Model tracking and champion selection

CLI Scripts:
- register_model.py: Add entries to the registry
- select_best.py: Pick champions by metric
"""

from .evaluate_model import (
    evaluate_model,
    evaluate_magnitude,
    evaluate_direction,
    evaluate_model_simple,
    aggregate_fold_results,
    compare_models,
    generate_evaluation_summary,
    MagnitudeMetrics,
    DirectionMetrics,
    EvaluationResult
)

from .leakage_checks import (
    LeakageChecker,
    validate_no_leakage,
    assert_temporal_integrity
)

from .error_analysis import (
    plot_prediction_vs_actual,
    plot_residuals_by_ssi_decile,
    plot_lift_chart,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_time_series_predictions,
    generate_error_analysis_report
)

from .model_registry import (
    ModelRegistry,
    ModelSpec
)

__version__ = '1.0.0'
__all__ = [
    # Evaluation
    'evaluate_model',
    'evaluate_magnitude',
    'evaluate_direction',
    'evaluate_model_simple',
    'aggregate_fold_results',
    'compare_models',
    'generate_evaluation_summary',
    'MagnitudeMetrics',
    'DirectionMetrics',
    'EvaluationResult',
    
    # Leakage
    'LeakageChecker',
    'validate_no_leakage',
    'assert_temporal_integrity',
    
    # Plots
    'plot_prediction_vs_actual',
    'plot_residuals_by_ssi_decile',
    'plot_lift_chart',
    'plot_confusion_matrix',
    'plot_roc_curve',
    'plot_time_series_predictions',
    'generate_error_analysis_report',
    
    # Registry
    'ModelRegistry',
    'ModelSpec'
]
