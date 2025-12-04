"""
Error Analysis and Visualization
Produces diagnostic plots for model evaluation.

Issue #17: Unified Evaluation and Model Registry
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve
import warnings
warnings.filterwarnings('ignore')


# Set style
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = {
    'primary': '#2563eb',
    'secondary': '#10b981', 
    'negative': '#ef4444',
    'neutral': '#6b7280',
    'accent': '#f59e0b'
}


def plot_prediction_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    commodity: str = "",
    model_name: str = "",
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Scatter plot of predicted vs actual returns.
    
    Args:
        y_true: Actual return values
        y_pred: Predicted return values
        commodity: Commodity name for title
        model_name: Model name for title
        save_path: Optional path to save figure
    
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Scatter plot
    ax.scatter(y_true, y_pred, alpha=0.5, c=COLORS['primary'], s=30)
    
    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    # Zero lines
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
    
    # Labels
    ax.set_xlabel('Actual Return', fontsize=12)
    ax.set_ylabel('Predicted Return', fontsize=12)
    ax.set_title(f'Prediction vs Actual - {commodity} ({model_name})', fontsize=14)
    ax.legend()
    
    # Stats annotation
    corr = np.corrcoef(y_true, y_pred)[0, 1]
    mae = np.mean(np.abs(y_true - y_pred))
    textstr = f'Correlation: {corr:.3f}\nMAE: {mae:.4f}'
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    
    return fig


def plot_residuals_by_ssi_decile(
    residuals: np.ndarray,
    ssi_values: np.ndarray,
    commodity: str = "",
    model_name: str = "",
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Box plot of residuals grouped by SSI deciles.
    
    Args:
        residuals: Prediction residuals (y_true - y_pred)
        ssi_values: SSI values for each prediction
        commodity: Commodity name for title
        model_name: Model name for title
        save_path: Optional path to save figure
    
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create deciles
    try:
        deciles = pd.qcut(ssi_values, q=10, labels=False, duplicates='drop')
    except ValueError:
        # If not enough unique values, use fewer bins
        deciles = pd.qcut(ssi_values, q=5, labels=False, duplicates='drop')
    
    # Group residuals by decile
    df = pd.DataFrame({'residual': residuals, 'decile': deciles})
    grouped = df.groupby('decile')['residual'].apply(list)
    
    # Box plot
    bp = ax.boxplot([grouped[d] for d in sorted(grouped.index)], 
                     patch_artist=True)
    
    # Color boxes
    for patch in bp['boxes']:
        patch.set_facecolor(COLORS['primary'])
        patch.set_alpha(0.6)
    
    # Labels
    ax.set_xlabel('SSI Decile', fontsize=12)
    ax.set_ylabel('Residual (Actual - Predicted)', fontsize=12)
    ax.set_title(f'Residuals by SSI Decile - {commodity} ({model_name})', fontsize=14)
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    
    # Decile labels
    ax.set_xticklabels([f'D{i+1}' for i in range(len(grouped))])
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    
    return fig


def plot_lift_chart(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    n_bins: int = 10,
    commodity: str = "",
    model_name: str = "",
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Lift chart for direction predictions.
    
    Shows how much better the model is at identifying positive cases
    compared to random selection, when sorted by predicted probability.
    
    Args:
        y_true: True direction labels (0/1)
        y_pred_proba: Predicted probabilities for class 1
        n_bins: Number of bins
        commodity: Commodity name for title
        model_name: Model name for title
        save_path: Optional path to save figure
    
    Returns:
        Matplotlib figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Sort by predicted probability descending
    sorted_indices = np.argsort(y_pred_proba)[::-1]
    y_true_sorted = y_true[sorted_indices]
    
    # Calculate cumulative stats
    n = len(y_true)
    baseline_rate = y_true.mean()
    
    decile_size = n // n_bins
    
    lift_values = []
    cumulative_gains = []
    
    for i in range(n_bins):
        start_idx = i * decile_size
        end_idx = (i + 1) * decile_size if i < n_bins - 1 else n
        
        decile_positives = y_true_sorted[start_idx:end_idx].sum()
        decile_size_actual = end_idx - start_idx
        decile_rate = decile_positives / decile_size_actual
        
        lift = decile_rate / baseline_rate if baseline_rate > 0 else 0
        lift_values.append(lift)
        
        # Cumulative gain
        cumulative_positives = y_true_sorted[:end_idx].sum()
        total_positives = y_true.sum()
        cumulative_gain = cumulative_positives / total_positives if total_positives > 0 else 0
        cumulative_gains.append(cumulative_gain)
    
    # Lift chart
    x = range(1, n_bins + 1)
    ax1.bar(x, lift_values, color=COLORS['primary'], alpha=0.7)
    ax1.axhline(y=1.0, color='red', linestyle='--', label='Baseline (random)')
    ax1.set_xlabel('Decile (by predicted probability)', fontsize=11)
    ax1.set_ylabel('Lift', fontsize=11)
    ax1.set_title(f'Lift Chart - {commodity}', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'D{i}' for i in x])
    ax1.legend()
    
    # Cumulative gains chart
    x_pct = [(i+1)/n_bins for i in range(n_bins)]
    ax2.plot(x_pct, cumulative_gains, 'o-', color=COLORS['primary'], 
             label='Model', markersize=8, linewidth=2)
    ax2.plot([0, 1], [0, 1], 'r--', label='Random')
    ax2.fill_between(x_pct, [0] + cumulative_gains[:-1], cumulative_gains, 
                     alpha=0.2, color=COLORS['primary'])
    ax2.set_xlabel('Proportion of Predictions', fontsize=11)
    ax2.set_ylabel('Cumulative Gain (% of positives captured)', fontsize=11)
    ax2.set_title(f'Cumulative Gains - {commodity}', fontsize=12)
    ax2.legend()
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    
    plt.suptitle(f'{model_name}', fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    
    return fig


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    commodity: str = "",
    model_name: str = "",
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot confusion matrix for direction predictions.
    
    Args:
        y_true: True direction labels (0/1)
        y_pred: Predicted direction labels (0/1)
        commodity: Commodity name for title
        model_name: Model name for title
        save_path: Optional path to save figure
    
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalize
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Plot
    im = ax.imshow(cm_normalized, interpolation='nearest', cmap='Blues')
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Proportion', rotation=-90, va="bottom")
    
    # Labels
    classes = ['Down (0)', 'Up (1)']
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           ylabel='Actual',
           xlabel='Predicted')
    
    # Add text annotations
    thresh = cm_normalized.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f'{cm[i, j]}\n({cm_normalized[i, j]:.1%})',
                    ha="center", va="center",
                    color="white" if cm_normalized[i, j] > thresh else "black",
                    fontsize=12)
    
    ax.set_title(f'Confusion Matrix - {commodity} ({model_name})', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    
    return fig


def plot_roc_curve(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    commodity: str = "",
    model_name: str = "",
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot ROC curve for direction predictions.
    
    Args:
        y_true: True direction labels (0/1)
        y_pred_proba: Predicted probabilities for class 1
        commodity: Commodity name for title
        model_name: Model name for title
        save_path: Optional path to save figure
    
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    
    # Calculate AUC
    from sklearn.metrics import auc
    roc_auc = auc(fpr, tpr)
    
    # Plot ROC curve
    ax.plot(fpr, tpr, color=COLORS['primary'], lw=2, 
            label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random')
    
    # Fill area
    ax.fill_between(fpr, tpr, alpha=0.2, color=COLORS['primary'])
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(f'ROC Curve - {commodity} ({model_name})', fontsize=14)
    ax.legend(loc="lower right")
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    
    return fig


def plot_time_series_predictions(
    dates: pd.Series,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    commodity: str = "",
    model_name: str = "",
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Time series plot of actual vs predicted returns.
    
    Args:
        dates: Date series
        y_true: Actual return values
        y_pred: Predicted return values
        commodity: Commodity name for title
        model_name: Model name for title
        save_path: Optional path to save figure
    
    Returns:
        Matplotlib figure
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    
    dates = pd.to_datetime(dates)
    
    # Actual vs Predicted
    ax1.plot(dates, y_true, label='Actual', color=COLORS['primary'], alpha=0.8, linewidth=1.5)
    ax1.plot(dates, y_pred, label='Predicted', color=COLORS['secondary'], alpha=0.8, linewidth=1.5)
    ax1.fill_between(dates, y_true, y_pred, alpha=0.2, color=COLORS['accent'])
    ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax1.set_ylabel('Return', fontsize=11)
    ax1.set_title(f'Actual vs Predicted Returns - {commodity} ({model_name})', fontsize=14)
    ax1.legend(loc='upper left')
    
    # Residuals
    residuals = y_true - y_pred
    colors = [COLORS['secondary'] if r >= 0 else COLORS['negative'] for r in residuals]
    ax2.bar(dates, residuals, color=colors, alpha=0.7, width=5)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_xlabel('Date', fontsize=11)
    ax2.set_ylabel('Residual', fontsize=11)
    ax2.set_title('Prediction Residuals Over Time', fontsize=12)
    
    # Format x-axis
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    
    return fig


def generate_error_analysis_report(
    predictions_df: pd.DataFrame,
    commodity: str,
    model_name: str,
    output_dir: Path,
    ssi_col: Optional[str] = None
) -> Dict[str, Path]:
    """
    Generate all error analysis plots for a model.
    
    Args:
        predictions_df: DataFrame with columns:
            - date: Date of prediction
            - actual_return: True return
            - predicted_return: Predicted return
            - actual_direction: True direction (0/1)
            - predicted_direction: Predicted direction (0/1)
            - predicted_proba: Predicted probability
            - ssi (optional): SSI values
        commodity: Commodity name
        model_name: Model name
        output_dir: Directory to save plots
        ssi_col: Optional column name for SSI values
    
    Returns:
        Dictionary mapping plot names to file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📊 Generating Error Analysis for {commodity} - {model_name}")
    
    saved_plots = {}
    
    # 1. Prediction vs Actual
    fig1_path = output_dir / f'{commodity}_{model_name}_pred_vs_actual.png'
    plot_prediction_vs_actual(
        predictions_df['actual_return'].values,
        predictions_df['predicted_return'].values,
        commodity=commodity,
        model_name=model_name,
        save_path=fig1_path
    )
    saved_plots['pred_vs_actual'] = fig1_path
    plt.close()
    
    # 2. Time series
    if 'date' in predictions_df.columns:
        fig2_path = output_dir / f'{commodity}_{model_name}_time_series.png'
        plot_time_series_predictions(
            predictions_df['date'],
            predictions_df['actual_return'].values,
            predictions_df['predicted_return'].values,
            commodity=commodity,
            model_name=model_name,
            save_path=fig2_path
        )
        saved_plots['time_series'] = fig2_path
        plt.close()
    
    # 3. Residuals by SSI decile (if SSI available)
    if ssi_col and ssi_col in predictions_df.columns:
        fig3_path = output_dir / f'{commodity}_{model_name}_residuals_ssi.png'
        residuals = predictions_df['actual_return'].values - predictions_df['predicted_return'].values
        plot_residuals_by_ssi_decile(
            residuals,
            predictions_df[ssi_col].values,
            commodity=commodity,
            model_name=model_name,
            save_path=fig3_path
        )
        saved_plots['residuals_ssi'] = fig3_path
        plt.close()
    
    # 4. Lift chart (if probability available)
    if 'predicted_proba' in predictions_df.columns:
        fig4_path = output_dir / f'{commodity}_{model_name}_lift_chart.png'
        plot_lift_chart(
            predictions_df['actual_direction'].values,
            predictions_df['predicted_proba'].values,
            commodity=commodity,
            model_name=model_name,
            save_path=fig4_path
        )
        saved_plots['lift_chart'] = fig4_path
        plt.close()
    
    # 5. Confusion matrix
    fig5_path = output_dir / f'{commodity}_{model_name}_confusion_matrix.png'
    plot_confusion_matrix(
        predictions_df['actual_direction'].values,
        predictions_df['predicted_direction'].values,
        commodity=commodity,
        model_name=model_name,
        save_path=fig5_path
    )
    saved_plots['confusion_matrix'] = fig5_path
    plt.close()
    
    # 6. ROC curve (if probability available)
    if 'predicted_proba' in predictions_df.columns:
        # Check if we have both classes
        if len(np.unique(predictions_df['actual_direction'])) > 1:
            fig6_path = output_dir / f'{commodity}_{model_name}_roc_curve.png'
            plot_roc_curve(
                predictions_df['actual_direction'].values,
                predictions_df['predicted_proba'].values,
                commodity=commodity,
                model_name=model_name,
                save_path=fig6_path
            )
            saved_plots['roc_curve'] = fig6_path
            plt.close()
    
    print(f"  ✓ Generated {len(saved_plots)} plots")
    
    return saved_plots


# ============================================================================
# Demo
# ============================================================================

if __name__ == '__main__':
    print("="*80)
    print("ERROR ANALYSIS DEMO")
    print("="*80)
    
    # Generate sample data
    np.random.seed(42)
    n = 200
    
    dates = pd.date_range('2023-01-01', periods=n, freq='W')
    y_true_return = np.random.randn(n) * 0.05
    y_pred_return = y_true_return + np.random.randn(n) * 0.02
    
    y_true_direction = (y_true_return > 0).astype(int)
    y_pred_proba = 1 / (1 + np.exp(-y_pred_return * 20))
    y_pred_direction = (y_pred_proba > 0.5).astype(int)
    
    ssi_values = np.random.randn(n)
    
    predictions_df = pd.DataFrame({
        'date': dates,
        'actual_return': y_true_return,
        'predicted_return': y_pred_return,
        'actual_direction': y_true_direction,
        'predicted_direction': y_pred_direction,
        'predicted_proba': y_pred_proba,
        'ssi': ssi_values
    })
    
    # Generate report
    output_dir = Path('demo_plots')
    saved_plots = generate_error_analysis_report(
        predictions_df,
        commodity='Henry_Hub',
        model_name='XGBoost_Demo',
        output_dir=output_dir,
        ssi_col='ssi'
    )
    
    print(f"\n✅ Demo complete! Plots saved to: {output_dir}")
    for name, path in saved_plots.items():
        print(f"  - {name}: {path}")
