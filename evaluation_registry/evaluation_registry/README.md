# Unified Evaluation and Model Registry

**Issue #17** - Common evaluation pipeline and lightweight model registry for consistent comparison and reproducible selection.

## Components

### 1. `evaluate_model.py` - Common Evaluator

Function `evaluate_model(preds, truth)` that returns:

**Magnitude Metrics:**
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R² (Coefficient of Determination)
- MAPE (Mean Absolute Percentage Error)
- Directional Accuracy (from magnitude predictions)

**Direction Metrics:**
- Accuracy
- F1 Score
- Precision / Recall
- ROC-AUC
- PR-AUC (Average Precision)
- Brier Score
- Log Loss
- Hit Ratio

```python
from evaluate_model import evaluate_model

result = evaluate_model(
    y_true_return=actual_returns,
    y_pred_return=predicted_returns,
    y_true_direction=actual_direction,
    y_pred_direction=predicted_direction,
    y_pred_direction_proba=predicted_probabilities,
    commodity="Henry_Hub",
    model_name="XGBoost"
)

print(f"MAE: {result.magnitude_metrics.mae:.4f}")
print(f"Accuracy: {result.direction_metrics.accuracy:.4f}")
```

### 2. `leakage_checks.py` - Leakage Validation

Validates no target or future feature leakage:

```python
from leakage_checks import LeakageChecker, validate_no_leakage

checker = LeakageChecker()

# Check train/test temporal integrity
result = checker.check_train_test_overlap(train_df, test_df, date_col='date')

# Check for suspiciously correlated features
result = checker.check_target_leakage(df, target_col='return_next', feature_cols=features)

# Assert: all feature timestamps <= target timestamp - 1 week
from leakage_checks import assert_temporal_integrity
assert_temporal_integrity(train_dates, test_dates, gap_weeks=1)
```

### 3. `error_analysis.py` - Diagnostic Plots

Produces plots:
- **Prediction vs Actual**: Scatter plot with correlation
- **Residuals by SSI Decile**: Box plots showing errors by SSI level
- **Lift Chart**: Model lift over random baseline
- **Cumulative Gains**: ROC-style curve for direction
- **Confusion Matrix**: Direction classification breakdown
- **ROC Curve**: With AUC annotation
- **Time Series**: Actual vs predicted over time

```python
from error_analysis import generate_error_analysis_report

saved_plots = generate_error_analysis_report(
    predictions_df=df,
    commodity="Henry_Hub",
    model_name="XGBoost",
    output_dir="reports/plots",
    ssi_col="ssi"  # Optional
)
```

### 4. `model_registry.py` - Model Registry

YAML/JSON spec per trained model with:
- Model name, version
- Training data span
- Feature list hash
- Metrics
- Artifact paths
- Git commit

```python
from model_registry import ModelRegistry

registry = ModelRegistry('models/registry')

# Register a model
spec = registry.register_model(
    model_name='xgb',
    version='v1',
    commodity='Henry_Hub',
    target='both',
    training_data_start='2022-01-01',
    training_data_end='2024-06-30',
    n_training_samples=500,
    feature_list=['feat1', 'feat2'],
    metrics={'mae': 0.025, 'accuracy': 0.58},
    artifact_paths={'model': 'artifacts/model.joblib'}
)

# Select best model
best = registry.select_best('Henry_Hub', metric='accuracy', set_as_champion=True)

# Get champion
champion = registry.get_champion('Henry_Hub')
```

## CLI Scripts

### `register_model.py` - Add Registry Entries

```bash
# With metrics file
python register_model.py --commodity Henry_Hub --model xgb --version v1 \
    --metrics-file results/metrics.json --artifact model.joblib

# With inline metrics
python register_model.py --commodity Brent --model ridge --version v1 \
    --metric mae=0.025 --metric accuracy=0.58 --set-champion
```

### `select_best.py` - Pick Champions

```bash
# Find best for one commodity
python select_best.py --commodity Henry_Hub --metric accuracy

# Find and set champions for all commodities
python select_best.py --all --metric roc_auc --set-champion

# Compare all models
python select_best.py --all --compare
```

## Integration with Training Pipeline

Run the full evaluation pipeline after training:

```bash
python run_evaluation.py \
    --predictions-file cleaned_data/xgboost_artifacts/xgb_predictions.csv \
    --model-name xgb \
    --version v1 \
    --output-dir reports \
    --register \
    --set-champion
```

This will:
1. ✅ Evaluate all commodities and folds
2. ✅ Run leakage checks
3. ✅ Generate diagnostic plots
4. ✅ Export standard `oos_predictions.csv`
5. ✅ Register models and set champions

## Standard Output Format

`reports/oos_predictions.csv` with columns:
- `commodity`: Ticker symbol
- `week_start`: Prediction week start date
- `y_true`: Actual return
- `y_pred_ret`: Predicted return
- `y_pred_dir_prob`: Predicted direction probability
- `model_name`: Model identifier

## Registry Structure

```
models/registry/
├── registry.yaml          # Main index
├── models/
│   ├── Henry_Hub/
│   │   ├── xgb_v1.yaml
│   │   ├── xgb_v2.yaml
│   │   └── champion.yaml
│   └── Brent/
│       └── ...
└── artifacts/
```

## Quick Start

1. **After training**, run evaluation:
   ```bash
   python run_evaluation.py --predictions-file xgb_predictions.csv --register
   ```

2. **Compare models**:
   ```bash
   python select_best.py --all --compare
   ```

3. **Set champions**:
   ```bash
   python select_best.py --all --metric accuracy --set-champion
   ```

## Requirements

```
numpy
pandas
scikit-learn
matplotlib
pyyaml
```

## Integration Example

Add to your `train_xgboost_model.py`:

```python
# At the end of training
from run_evaluation import run_evaluation, register_models

# Run evaluation
results = run_evaluation(predictions_df, model_name='xgb')

# Register with version based on date
version = f"v{datetime.now().strftime('%Y%m%d')}"
register_models(results, 'xgb', version, set_champion=True)
```
