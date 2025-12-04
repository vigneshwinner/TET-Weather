"""
Random Forest Model Training (FAST VERSION)
Non-parametric benchmark for next-week commodity returns (magnitude + direction)

- Uses the same feature matrix and walk-forward CV scheme as:
    - Issue 11: baseline_ridge_model.py
    - Issue 12: train_xgboost_model.py

- Trains:
    * RandomForestRegressor for return_next (magnitude)
    * RandomForestClassifier for direction_next (direction)

- FAST SETTINGS:
    * Very small hyperparameter grid
    * Limit number of folds (test weeks) per commodity
    * Fewer permutation-importance repeats

- Still:
    * Reports per-fold variance of key scores (R^2, accuracy)
    * Logs permutation importances on the last fold for each commodity.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
)
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance

import warnings
warnings.filterwarnings("ignore")

print("=" * 80)
print("RANDOM FOREST MODEL TRAINING - FAST NON-PARAMETRIC BENCHMARK")
print("=" * 80)

# ============================================================================
# Configuration
# ============================================================================

# For reference / consistency with XGBoost script
TRAIN_START = "2022-12-05"
FIRST_TEST = "2024-01-01"
RANDOM_STATE = 42

COMMODITIES = ["Brent", "Henry_Hub", "Power", "Copper", "Corn"]

# ---- Speed controls ----
# Max test folds (weeks) per commodity; set to None for all
MAX_FOLDS = 25           # was: all test weeks (~100+)
# Fewer permutation importance repeats
N_PERM_REPEATS = 5       # was: 30

OUTPUT_DIR = Path("cleaned_data/random_forest_artifacts")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("\n⚙️  Configuration:")
print(f"  TRAIN_START      : {TRAIN_START}")
print(f"  FIRST_TEST       : {FIRST_TEST}")
print(f"  COMMODITIES      : {COMMODITIES}")
print(f"  MAX_FOLDS        : {MAX_FOLDS}")
print(f"  N_PERM_REPEATS   : {N_PERM_REPEATS}")
print(f"  OUTPUT_DIR       : {OUTPUT_DIR.resolve()}")

# ---- SMALL hyperparameter grid (fast) ----
# Drastically smaller than original (48 combos).
RF_PARAM_GRID = {
    "n_estimators": [200],          # single value (no search over tree count)
    "max_depth": [10],             # one reasonable finite depth
    "min_samples_split": [2, 5],   # tiny variation
    "min_samples_leaf": [1],       # single value
    "max_features": ["sqrt"],      # standard RF setting
}


def generate_param_grid(grid_dict):
    """Generate a list of param dicts from a small grid."""
    from itertools import product

    keys = list(grid_dict.keys())
    values = [grid_dict[k] for k in keys]
    param_list = []
    for combo in product(*values):
        params = dict(zip(keys, combo))
        param_list.append(params)
    return param_list


RF_PARAM_LIST = generate_param_grid(RF_PARAM_GRID)
print(f"  RF grid size     : {len(RF_PARAM_LIST)} combinations")

# ============================================================================
# 1. Load price data & construct targets (same as XGBoost / Ridge)
# ============================================================================

print("\n📂 Loading price data and constructing targets...")

price_files = {
    "Brent": "cleaned_data/Brent_3yr.csv",
    "Henry_Hub": "cleaned_data/Henry_Hub_3yr.csv",
    "Power": "cleaned_data/Power_3yr.csv",
    "Copper": "cleaned_data/Copper_3yr.csv",
    "Corn": "cleaned_data/Corn_3yr.csv",
}

returns_dfs = []

for commodity, filepath in price_files.items():
    df = pd.read_csv(filepath, skiprows=2)
    df.columns = ["Date", "Open", "High", "Low", "Close", "Volume"]
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # Log returns and next-week targets
    df["return"] = np.log(df["Close"] / df["Close"].shift(1))
    df["return_next"] = df["return"].shift(-1)
    df["direction_next"] = (df["return_next"] > 0).astype(int)

    # Week anchored to Monday (same convention as other scripts)
    df["week"] = df["Date"] - pd.to_timedelta(df["Date"].dt.dayofweek.astype("int64"), unit="D")

    df["commodity"] = commodity
    df = df[["Date", "week", "commodity", "Close", "return", "return_next", "direction_next"]].copy()

    returns_dfs.append(df)
    print(f"  ✓ {commodity}: {len(df)} records")

returns_df = pd.concat(returns_dfs, ignore_index=True)

# ============================================================================
# 2. Load feature data and assemble weekly feature matrix
# ============================================================================

print("\n📂 Loading feature data...")

interactions_df = pd.read_csv("cleaned_data/weather_eia_interactions.csv")
interactions_df["date"] = pd.to_datetime(interactions_df["date"])
interactions_df["week"] = interactions_df["date"] - pd.to_timedelta(
    interactions_df["date"].dt.dayofweek.astype("int64"), unit="D"
)

degree_days_df = pd.read_csv("cleaned_data/degree_days.csv")
degree_days_df["week"] = pd.to_datetime(degree_days_df["week"])

commodity_map = {
    "Crude_Oil": "Brent",
    "Natural_Gas": "Henry_Hub",
    "Power": "Power",
    "Copper": "Copper",
    "Corn": "Corn",
}
interactions_df["commodity_mapped"] = interactions_df["commodity"].map(commodity_map)

feature_cols = [
    col
    for col in interactions_df.columns
    if col not in ["date", "week", "region", "commodity", "commodity_mapped", "date_eia"]
]

interactions_weekly = (
    interactions_df.groupby(["week", "commodity_mapped"])[feature_cols]
    .mean()
    .reset_index()
)
interactions_weekly = interactions_weekly.rename(columns={"commodity_mapped": "commodity"})
interactions_weekly = interactions_weekly[interactions_weekly["commodity"].notna()]

dd_weekly = (
    degree_days_df.groupby("week")
    .agg(
        {
            "hdd_weekly_sum": "mean",
            "cdd_weekly_sum": "mean",
            "hdd_7day_avg": "mean",
            "cdd_7day_avg": "mean",
            "hdd_14day_avg": "mean",
            "cdd_14day_avg": "mean",
            "hdd_30day_avg": "mean",
            "cdd_30day_avg": "mean",
        }
    )
    .reset_index()
)

merged_df = returns_df.copy()
merged_df = merged_df.merge(interactions_weekly, on=["week", "commodity"], how="left")
merged_df = merged_df.merge(dd_weekly, on="week", how="left")

print("  ℹ️  Commodities without weather-EIA interactions will use degree-days only.")

# Keep rows with defined next-week return
merged_df = merged_df[merged_df["return_next"].notna()].copy()

# Report missingness before filling
missing_by_commodity = merged_df.groupby("commodity").apply(lambda x: x.isnull().sum().sum())
for commodity, missing_count in missing_by_commodity.items():
    if missing_count > 0:
        print(f"    {commodity}: {missing_count} missing values (will fill with 0)")

merged_df = merged_df.fillna(0)

exclude_cols = ["Date", "week", "commodity", "Close", "return", "return_next", "direction_next"]
feature_columns = [col for col in merged_df.columns if col not in exclude_cols]

print(f"  ✓ Total features: {len(feature_columns)}")
print(f"  ✓ Total records : {len(merged_df)}")

# ============================================================================
# 3. Walk-forward CV and Random Forest training (FAST)
# ============================================================================

results_list = []
predictions_list = []
best_params_summary = {}
perm_importance_summary = {}

first_test_week = pd.to_datetime(FIRST_TEST) - pd.to_timedelta(
    pd.to_datetime(FIRST_TEST).dayofweek, unit="D"
)

for commodity in COMMODITIES:
    print("\n" + "=" * 80)
    print(f"COMMODITY: {commodity}")
    print("=" * 80)

    commodity_df = merged_df[merged_df["commodity"] == commodity].copy()
    commodity_df = commodity_df.sort_values("Date").reset_index(drop=True)

    print(f"\n📊 Data: {len(commodity_df)} records")
    print(
        f"  Date range: {commodity_df['Date'].min().date()} to {commodity_df['Date'].max().date()}"
    )

    weeks = sorted(commodity_df["week"].unique())
    test_weeks = [w for w in weeks if w >= first_test_week]

    # Limit number of folds for speed
    if MAX_FOLDS is not None:
        test_weeks = test_weeks[:MAX_FOLDS]

    print(f"  First test week: {first_test_week.date()}")
    print(f"  Number of test weeks (folds): {len(test_weeks)}")

    fold_index = 0
    best_params_summary[commodity] = {"regression": [], "classification": []}
    perm_importance_summary[commodity] = {"regression": None, "classification": None}

    fold_r2_scores = []
    fold_acc_scores = []

    last_fold_artifacts = None

    for test_week in test_weeks:
        fold_index += 1
        print(f"\n--- Fold {fold_index} | Test week starting {test_week.date()} ---")

        train_mask = commodity_df["week"] < test_week
        test_mask = commodity_df["week"] == test_week

        train_df = commodity_df[train_mask].copy()
        test_df = commodity_df[test_mask].copy()

        if len(train_df) < 50 or len(test_df) == 0:
            print("  ⚠️  Skipping fold (insufficient train or test size)")
            continue

        # Train/val split: last 20% of train as validation (time-ordered)
        split_idx = int(len(train_df) * 0.8)
        train_core_df = train_df.iloc[:split_idx]
        val_df = train_df.iloc[split_idx:]

        X_train = train_core_df[feature_columns].values
        y_train_reg = train_core_df["return_next"].values
        y_train_clf = train_core_df["direction_next"].values

        X_val = val_df[feature_columns].values
        y_val_reg = val_df["return_next"].values
        y_val_clf = val_df["direction_next"].values

        X_train_full = train_df[feature_columns].values
        y_train_reg_full = train_df["return_next"].values
        y_train_clf_full = train_df["direction_next"].values

        X_test = test_df[feature_columns].values
        y_test_reg = test_df["return_next"].values
        y_test_clf = test_df["direction_next"].values

        # Standardize (kept for consistency with XGBoost pipeline, though RF doesn't require it)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_train_full_scaled = scaler.transform(X_train_full)
        X_test_scaled = scaler.transform(X_test)

        # ------------------------------
        # 3a. Regression RF - hyperparam search (tiny grid)
        # ------------------------------
        best_reg_mae = np.inf
        best_reg_params = None

        for params in RF_PARAM_LIST:
            rf_reg = RandomForestRegressor(
                random_state=RANDOM_STATE,
                n_jobs=-1,
                **params,
            )
            rf_reg.fit(X_train_scaled, y_train_reg)
            y_val_pred = rf_reg.predict(X_val_scaled)
            mae_val = mean_absolute_error(y_val_reg, y_val_pred)

            if mae_val < best_reg_mae:
                best_reg_mae = mae_val
                best_reg_params = params

        print(f"  ✓ Best RF regression params (val MAE={best_reg_mae:.6f}): {best_reg_params}")

        # Retrain regression model on full training (train + val)
        rf_reg_final = RandomForestRegressor(
            random_state=RANDOM_STATE,
            n_jobs=-1,
            **best_reg_params,
        )
        rf_reg_final.fit(X_train_full_scaled, y_train_reg_full)

        # Evaluate on test
        y_pred_reg = rf_reg_final.predict(X_test_scaled)
        reg_mae = mean_absolute_error(y_test_reg, y_pred_reg)
        reg_rmse = np.sqrt(mean_squared_error(y_test_reg, y_pred_reg))
        reg_r2 = r2_score(y_test_reg, y_pred_reg)

        fold_r2_scores.append(reg_r2)

        # ------------------------------
        # 3b. Classification RF - hyperparam search (tiny grid)
        # ------------------------------
        best_clf_metric = -np.inf
        best_clf_params = None

        for params in RF_PARAM_LIST:
            rf_clf = RandomForestClassifier(
                random_state=RANDOM_STATE,
                n_jobs=-1,
                class_weight="balanced",
                **params,
            )
            rf_clf.fit(X_train_scaled, y_train_clf)
            y_val_proba = rf_clf.predict_proba(X_val_scaled)[:, 1]
            # Use ROC-AUC on validation as objective (when possible)
            try:
                roc_val = roc_auc_score(y_val_clf, y_val_proba)
            except Exception:
                roc_val = np.nan

            if np.isnan(roc_val):
                continue

            if roc_val > best_clf_metric:
                best_clf_metric = roc_val
                best_clf_params = params

        if best_clf_params is None:
            # Fallback: simple default RF if ROC-AUC could not be computed
            print("  ⚠️  ROC-AUC undefined in validation; using default RF params.")
            best_clf_params = {
                "n_estimators": 200,
                "max_depth": 10,
                "min_samples_split": 2,
                "min_samples_leaf": 1,
                "max_features": "sqrt",
            }
        else:
            print(f"  ✓ Best RF classifier params (val ROC-AUC={best_clf_metric:.6f}): {best_clf_params}")

        # Retrain classification model on full training
        rf_clf_final = RandomForestClassifier(
            random_state=RANDOM_STATE,
            n_jobs=-1,
            class_weight="balanced",
            **best_clf_params,
        )
        rf_clf_final.fit(X_train_full_scaled, y_train_clf_full)

        # Evaluate classifier on test
        y_pred_clf = rf_clf_final.predict(X_test_scaled)
        y_pred_clf_proba = rf_clf_final.predict_proba(X_test_scaled)[:, 1]
        y_pred_dir_from_reg = (y_pred_reg > 0).astype(int)

        clf_accuracy = accuracy_score(y_test_clf, y_pred_clf)
        clf_f1 = f1_score(y_test_clf, y_pred_clf, zero_division=0)

        try:
            clf_roc_auc = roc_auc_score(y_test_clf, y_pred_clf_proba)
        except Exception:
            clf_roc_auc = np.nan

        try:
            clf_pr_auc = average_precision_score(y_test_clf, y_pred_clf_proba)
        except Exception:
            clf_pr_auc = np.nan

        try:
            clf_brier = brier_score_loss(y_test_clf, y_pred_clf_proba)
        except Exception:
            clf_brier = np.nan

        try:
            clf_logloss = log_loss(y_test_clf, y_pred_clf_proba)
        except Exception:
            clf_logloss = np.nan

        dir_hit_from_reg = accuracy_score(y_test_clf, y_pred_dir_from_reg)

        fold_acc_scores.append(clf_accuracy)

        # Store metrics
        results_list.append(
            {
                "commodity": commodity,
                "fold_index": fold_index,
                "test_week": test_week.strftime("%Y-%m-%d"),
                "n_train": len(train_df),
                "n_test": len(test_df),
                "reg_mae": reg_mae,
                "reg_rmse": reg_rmse,
                "reg_r2": reg_r2,
                "clf_accuracy": clf_accuracy,
                "clf_f1": clf_f1,
                "clf_roc_auc": clf_roc_auc,
                "clf_pr_auc": clf_pr_auc,
                "clf_brier": clf_brier,
                "clf_logloss": clf_logloss,
                "dir_hit_from_reg": dir_hit_from_reg,
            }
        )

        # Store predictions (per-row)
        for i, row in test_df.iterrows():
            idx = list(test_df.index).index(i)
            predictions_list.append(
                {
                    "commodity": commodity,
                    "Date": row["Date"].strftime("%Y-%m-%d"),
                    "week": row["week"].strftime("%Y-%m-%d"),
                    "return_next_true": row["return_next"],
                    "return_next_pred_rf": float(y_pred_reg[idx]),
                    "direction_true": int(row["direction_next"]),
                    "direction_pred_rf": int(y_pred_clf[idx]),
                    "direction_pred_proba_rf": float(y_pred_clf_proba[idx]),
                }
            )

        # Save per-fold artifacts
        artifact = {
            "commodity": commodity,
            "fold_index": fold_index,
            "test_week": test_week,
            "reg_model": rf_reg_final,
            "clf_model": rf_clf_final,
            "scaler": scaler,
            "feature_names": feature_columns,
            "reg_params": best_reg_params,
            "clf_params": best_clf_params,
        }

        artifact_path = OUTPUT_DIR / f"{commodity}_fold{fold_index}_rf.joblib"
        joblib.dump(artifact, artifact_path)

        last_fold_artifacts = {
            "artifact": artifact,
            "X_test_scaled": X_test_scaled,
            "y_test_reg": y_test_reg,
            "y_test_clf": y_test_clf,
        }

        # Track best params
        best_params_summary[commodity]["regression"].append(
            {
                "fold_index": fold_index,
                "test_week": test_week.strftime("%Y-%m-%d"),
                "params": best_reg_params,
                "val_mae": best_reg_mae,
            }
        )
        best_params_summary[commodity]["classification"].append(
            {
                "fold_index": fold_index,
                "test_week": test_week.strftime("%Y-%m-%d"),
                "params": best_clf_params,
                "val_metric": best_clf_metric,
            }
        )

    # After all folds for this commodity: analyze score variance & permutation importance
    if fold_r2_scores:
        r2_var = float(np.var(fold_r2_scores))
        print(f"\n📈 {commodity} - R² variance across folds: {r2_var:.6f}")
    else:
        r2_var = None
        print(f"\n⚠️  {commodity} - No valid R² scores to compute variance.")

    if fold_acc_scores:
        acc_var = float(np.var(fold_acc_scores))
        print(f"📈 {commodity} - Accuracy variance across folds: {acc_var:.6f}")
    else:
        acc_var = None
        print(f"⚠️  {commodity} - No valid accuracy scores to compute variance.")

    # Permutation importance on last fold (if any)
    if last_fold_artifacts is not None:
        artifact = last_fold_artifacts["artifact"]
        X_test_scaled = last_fold_artifacts["X_test_scaled"]
        y_test_reg = last_fold_artifacts["y_test_reg"]
        y_test_clf = last_fold_artifacts["y_test_clf"]

        feature_names = artifact["feature_names"]
        reg_model = artifact["reg_model"]
        clf_model = artifact["clf_model"]

        print(f"\n🔍 Computing permutation importance on last fold for {commodity}...")

        # Regression permutation importance
        reg_perm = permutation_importance(
            reg_model,
            X_test_scaled,
            y_test_reg,
            n_repeats=N_PERM_REPEATS,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        reg_importances = []
        order_reg = np.argsort(reg_perm.importances_mean)[::-1]
        for idx in order_reg[:20]:
            reg_importances.append(
                {
                    "feature": feature_names[idx],
                    "importance_mean": float(reg_perm.importances_mean[idx]),
                    "importance_std": float(reg_perm.importances_std[idx]),
                }
            )

        # Classification permutation importance
        clf_perm = permutation_importance(
            clf_model,
            X_test_scaled,
            y_test_clf,
            n_repeats=N_PERM_REPEATS,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        clf_importances = []
        order_clf = np.argsort(clf_perm.importances_mean)[::-1]
        for idx in order_clf[:20]:
            clf_importances.append(
                {
                    "feature": feature_names[idx],
                    "importance_mean": float(clf_perm.importances_mean[idx]),
                    "importance_std": float(clf_perm.importances_std[idx]),
                }
            )

        perm_importance_summary[commodity] = {
            "regression": reg_importances,
            "classification": clf_importances,
            "r2_variance": r2_var,
            "accuracy_variance": acc_var,
        }
    else:
        perm_importance_summary[commodity] = {
            "regression": None,
            "classification": None,
            "r2_variance": r2_var,
            "accuracy_variance": acc_var,
        }

# ============================================================================
# 4. Save results and summaries
# ============================================================================

print("\n💾 Saving Random Forest results and artifacts...")

results_df = pd.DataFrame(results_list)
predictions_df = pd.DataFrame(predictions_list)

results_path = OUTPUT_DIR / "rf_fold_results.csv"
predictions_path = OUTPUT_DIR / "rf_predictions.csv"
best_params_path = OUTPUT_DIR / "rf_best_params.json"
importance_path = OUTPUT_DIR / "rf_permutation_importance.json"

results_df.to_csv(results_path, index=False)
predictions_df.to_csv(predictions_path, index=False)

with open(best_params_path, "w") as f:
    json.dump(best_params_summary, f, indent=2)

with open(importance_path, "w") as f:
    json.dump(perm_importance_summary, f, indent=2)

print(f"  ✓ Fold results       -> {results_path}")
print(f"  ✓ Predictions        -> {predictions_path}")
print(f"  ✓ Best RF params     -> {best_params_path}")
print(f"  ✓ Permutation import -> {importance_path}")

# High-level summary
if not results_df.empty:
    print("\n📊 OVERALL RANDOM FOREST PERFORMANCE:")
    print(results_df.groupby("commodity")[["reg_mae", "reg_rmse", "reg_r2", "clf_accuracy"]].mean())
else:
    print("\n⚠️  No results were produced. Check data coverage and FIRST_TEST configuration.")

print("\n✅ Random Forest training complete.")
print("=" * 80)
