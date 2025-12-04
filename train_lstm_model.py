"""
LSTM Model Training (FAST MODE)
Prototype a simple LSTM to capture temporal dependencies using sliding windows.

FAST_MODE:
  - Single window length (L=12)
  - Limit folds per commodity
  - Smaller LSTM
  - Fewer epochs + aggressive early stopping
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler
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

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import warnings
warnings.filterwarnings("ignore")

print("=" * 80)
print("LSTM MODEL TRAINING - TEMPORAL BASELINE (FAST MODE)")
print("=" * 80)

# =============================================================================
# FAST MODE CONFIG
# =============================================================================

FAST_MODE = True  # set to False to revert to slower, more thorough training

TRAIN_START = "2022-12-05"
FIRST_TEST = "2024-01-01"
RANDOM_STATE = 42

COMMODITIES = ["Brent", "Henry_Hub", "Power", "Copper", "Corn"]

if FAST_MODE:
    WINDOW_LENGTHS = [12]      # single window length for speed
    HIDDEN_SIZE = 24           # smaller LSTM
    DROPOUT = 0.2
    BATCH_SIZE = 64            # larger batch
    MAX_EPOCHS = 20            # fewer epochs
    EARLY_STOP_PATIENCE = 3    # more aggressive early stopping
    MAX_FOLDS = 25             # cap number of folds (test weeks) per commodity
else:
    WINDOW_LENGTHS = [8, 12, 16]
    HIDDEN_SIZE = 32
    DROPOUT = 0.2
    BATCH_SIZE = 32
    MAX_EPOCHS = 50
    EARLY_STOP_PATIENCE = 7
    MAX_FOLDS = None  # use all folds

LEARNING_RATE = 1e-3
LAMBDA_DIR = 1.0  # weight for direction loss

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUTPUT_DIR = Path("cleaned_data/lstm_artifacts")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("\n⚙️  Configuration:")
print(f"  FAST_MODE        : {FAST_MODE}")
print(f"  TRAIN_START      : {TRAIN_START}")
print(f"  FIRST_TEST       : {FIRST_TEST}")
print(f"  COMMODITIES      : {COMMODITIES}")
print(f"  WINDOW_LENGTHS   : {WINDOW_LENGTHS}")
print(f"  HIDDEN_SIZE      : {HIDDEN_SIZE}")
print(f"  DROPOUT          : {DROPOUT}")
print(f"  BATCH_SIZE       : {BATCH_SIZE}")
print(f"  MAX_EPOCHS       : {MAX_EPOCHS}")
print(f"  EARLY_STOP       : {EARLY_STOP_PATIENCE}")
print(f"  MAX_FOLDS        : {MAX_FOLDS}")
print(f"  DEVICE           : {DEVICE}")
print(f"  OUTPUT_DIR       : {OUTPUT_DIR.resolve()}")

torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

# =============================================================================
# 1. Load price data & construct weekly returns/targets
# =============================================================================

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

    df["return"] = np.log(df["Close"] / df["Close"].shift(1))
    df["return_next"] = df["return"].shift(-1)
    df["direction_next"] = (df["return_next"] > 0).astype(int)

    df["week"] = df["Date"] - pd.to_timedelta(df["Date"].dt.dayofweek.astype("int64"), unit="D")
    df["commodity"] = commodity

    df = df[["Date", "week", "commodity", "Close", "return", "return_next", "direction_next"]].copy()
    returns_dfs.append(df)
    print(f"  ✓ {commodity}: {len(df)} records")

returns_df = pd.concat(returns_dfs, ignore_index=True)

# =============================================================================
# 2. Load feature data and assemble weekly feature matrix
# =============================================================================

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

merged_df = merged_df[merged_df["return_next"].notna()].copy()
missing_by_commodity = merged_df.groupby("commodity").apply(lambda x: x.isnull().sum().sum())
for commodity, missing_count in missing_by_commodity.items():
    if missing_count > 0:
        print(f"    {commodity}: {missing_count} missing values (will fill with 0)")

merged_df = merged_df.fillna(0)

exclude_cols = ["Date", "week", "commodity", "Close", "return", "return_next", "direction_next"]
feature_columns = [col for col in merged_df.columns if col not in exclude_cols]

print(f"  ✓ Total features: {len(feature_columns)}")
print(f"  ✓ Total records : {len(merged_df)}")

first_test_week = pd.to_datetime(FIRST_TEST) - pd.to_timedelta(
    pd.to_datetime(FIRST_TEST).dayofweek, unit="D"
)

# =============================================================================
# Dataset & Model
# =============================================================================

class SequenceDataset(Dataset):
    def __init__(self, X, y_reg, y_dir):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y_reg = torch.from_numpy(y_reg.astype(np.float32))
        self.y_dir = torch.from_numpy(y_dir.astype(np.float32))

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y_reg[idx], self.y_dir[idx]


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=32, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.reg_head = nn.Linear(hidden_size, 1)
        self.dir_head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        last_hidden = out[:, -1, :]
        h = self.dropout(last_hidden)
        reg_out = self.reg_head(h).squeeze(-1)
        dir_logits = self.dir_head(h).squeeze(-1)
        return reg_out, dir_logits


def build_windows_for_commodity(commodity_df, window_length):
    df = commodity_df.sort_values("Date").reset_index(drop=True).copy()

    features = df[feature_columns].values
    y_reg_full = df["return_next"].values
    y_dir_full = df["direction_next"].values
    weeks = df["week"].values

    T = features.shape[0]
    L = window_length

    X_list, y_reg_list, y_dir_list, last_week_list = [], [], [], []

    for i in range(L - 1, T):
        if np.isnan(y_reg_full[i]):
            continue
        window_feats = features[i - L + 1 : i + 1]
        X_list.append(window_feats)
        y_reg_list.append(y_reg_full[i])
        y_dir_list.append(y_dir_full[i])
        last_week_list.append(weeks[i])

    if not X_list:
        return (
            np.empty((0, L, features.shape[1])),
            np.array([]),
            np.array([]),
            np.array([]),
        )

    X_seq = np.stack(X_list, axis=0)
    y_reg = np.array(y_reg_list)
    y_dir = np.array(y_dir_list)
    last_weeks = np.array(last_week_list)

    return X_seq, y_reg, y_dir, last_weeks


def standardize_sequences(X_train, X_val, X_test):
    N_train, L, F = X_train.shape
    scaler = StandardScaler()
    scaler.fit(X_train.reshape(N_train * L, F))

    def transform(X):
        if X.shape[0] == 0:
            return X
        N, L_, F_ = X.shape
        X_flat = X.reshape(N * L_, F_)
        X_scaled = scaler.transform(X_flat)
        return X_scaled.reshape(N, L_, F_)

    return transform(X_train), transform(X_val), transform(X_test), scaler


# =============================================================================
# 3. Training Loop with Walk-forward CV (FAST)
# =============================================================================

results_list = []
predictions_list = []
artifact_index = []

bce_loss_fn = nn.BCEWithLogitsLoss()
mse_loss_fn = nn.MSELoss()

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

    for L in WINDOW_LENGTHS:
        print(f"\n--- Window length L = {L} weeks ---")

        X_all, y_reg_all, y_dir_all, last_weeks_all = build_windows_for_commodity(commodity_df, L)
        if X_all.shape[0] == 0:
            print("  ⚠️ No sequences for this window length. Skipping.")
            continue

        unique_weeks = sorted(np.unique(last_weeks_all))
        test_weeks = [w for w in unique_weeks if w >= np.datetime64(first_test_week)]

        if MAX_FOLDS is not None:
            test_weeks = test_weeks[:MAX_FOLDS]

        print(f"  Number of sequences: {len(X_all)}")
        print(f"  First test week: {first_test_week.date()}")
        print(f"  Number of weekly folds: {len(test_weeks)}")

        fold_index = 0
        fold_r2_scores = []
        fold_acc_scores = []

        for test_week in test_weeks:
            fold_index += 1
            print(f"\n>>> Fold {fold_index} | Test week starting {pd.to_datetime(test_week).date()}")

            train_mask = last_weeks_all < test_week
            test_mask = last_weeks_all == test_week

            X_train_all = X_all[train_mask]
            y_train_reg_all = y_reg_all[train_mask]
            y_train_dir_all = y_dir_all[train_mask]

            X_test = X_all[test_mask]
            y_test_reg = y_reg_all[test_mask]
            y_test_dir = y_dir_all[test_mask]

            if X_train_all.shape[0] < 40 or X_test.shape[0] == 0:
                print("  ⚠️ Skipping fold (insufficient train or test samples)")
                continue

            n_train_total = X_train_all.shape[0]
            split_idx = int(n_train_total * 0.8)
            X_train = X_train_all[:split_idx]
            y_train_reg = y_train_reg_all[:split_idx]
            y_train_dir = y_train_dir_all[:split_idx]

            X_val = X_train_all[split_idx:]
            y_val_reg = y_train_reg_all[split_idx:]
            y_val_dir = y_train_dir_all[split_idx:]

            X_train_scaled, X_val_scaled, X_test_scaled, scaler = standardize_sequences(
                X_train, X_val, X_test
            )

            train_ds = SequenceDataset(X_train_scaled, y_train_reg, y_train_dir)
            val_ds = SequenceDataset(X_val_scaled, y_val_reg, y_val_dir)
            test_ds = SequenceDataset(X_test_scaled, y_test_reg, y_test_dir)

            train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
            val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
            test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

            input_size = X_train_scaled.shape[2]
            model = LSTMModel(input_size=input_size, hidden_size=HIDDEN_SIZE, dropout=DROPOUT)
            model.to(DEVICE)

            optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

            best_val_loss = np.inf
            best_state_dict = None
            epochs_no_improve = 0

            for epoch in range(1, MAX_EPOCHS + 1):
                model.train()
                train_losses = []

                for batch_X, batch_y_reg, batch_y_dir in train_loader:
                    batch_X = batch_X.to(DEVICE)
                    batch_y_reg = batch_y_reg.to(DEVICE)
                    batch_y_dir = batch_y_dir.to(DEVICE)

                    optimizer.zero_grad()
                    pred_reg, pred_logits = model(batch_X)

                    loss_reg = mse_loss_fn(pred_reg, batch_y_reg)
                    loss_dir = bce_loss_fn(pred_logits, batch_y_dir)
                    loss = loss_reg + LAMBDA_DIR * loss_dir

                    loss.backward()
                    optimizer.step()
                    train_losses.append(loss.item())

                model.eval()
                val_losses = []
                with torch.no_grad():
                    for batch_X, batch_y_reg, batch_y_dir in val_loader:
                        batch_X = batch_X.to(DEVICE)
                        batch_y_reg = batch_y_reg.to(DEVICE)
                        batch_y_dir = batch_y_dir.to(DEVICE)

                        pred_reg, pred_logits = model(batch_X)
                        loss_reg = mse_loss_fn(pred_reg, batch_y_reg)
                        loss_dir = bce_loss_fn(pred_logits, batch_y_dir)
                        loss = loss_reg + LAMBDA_DIR * loss_dir
                        val_losses.append(loss.item())

                mean_train_loss = np.mean(train_losses) if train_losses else np.nan
                mean_val_loss = np.mean(val_losses) if val_losses else np.nan

                print(
                    f"  Epoch {epoch:02d} | train_loss={mean_train_loss:.5f} "
                    f"| val_loss={mean_val_loss:.5f}"
                )

                if mean_val_loss < best_val_loss - 1e-4:
                    best_val_loss = mean_val_loss
                    best_state_dict = model.state_dict()
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

                if epochs_no_improve >= EARLY_STOP_PATIENCE:
                    print(f"  Early stopping triggered at epoch {epoch}.")
                    break

            if best_state_dict is None:
                print("  ⚠️ No best model captured; skipping fold.")
                continue

            model.load_state_dict(best_state_dict)

            model.eval()
            all_y_reg_true, all_y_reg_pred = [], []
            all_y_dir_true, all_y_dir_pred, all_dir_proba = [], [], []

            start_time = time.perf_counter()
            with torch.no_grad():
                for batch_X, batch_y_reg, batch_y_dir in test_loader:
                    batch_X = batch_X.to(DEVICE)
                    batch_y_reg = batch_y_reg.to(DEVICE)
                    batch_y_dir = batch_y_dir.to(DEVICE)

                    pred_reg, pred_logits = model(batch_X)
                    pred_proba = torch.sigmoid(pred_logits)

                    all_y_reg_true.append(batch_y_reg.cpu().numpy())
                    all_y_reg_pred.append(pred_reg.cpu().numpy())
                    all_y_dir_true.append(batch_y_dir.cpu().numpy())
                    all_y_dir_pred.append((pred_proba.cpu().numpy() > 0.5).astype(int))
                    all_dir_proba.append(pred_proba.cpu().numpy())
            end_time = time.perf_counter()

            total_preds = len(y_test_reg)
            avg_inference_time = (end_time - start_time) / total_preds if total_preds > 0 else np.nan

            y_reg_true = np.concatenate(all_y_reg_true)
            y_reg_pred = np.concatenate(all_y_reg_pred)
            y_dir_true = np.concatenate(all_y_dir_true)
            y_dir_pred = np.concatenate(all_y_dir_pred)
            y_dir_proba = np.concatenate(all_dir_proba)

            reg_mae = mean_absolute_error(y_reg_true, y_reg_pred)
            reg_rmse = np.sqrt(mean_squared_error(y_reg_true, y_reg_pred))
            reg_r2 = r2_score(y_reg_true, y_reg_pred)
            fold_r2_scores.append(reg_r2)

            clf_accuracy = accuracy_score(y_dir_true, y_dir_pred)
            clf_f1 = f1_score(y_dir_true, y_dir_pred, zero_division=0)

            try:
                clf_roc_auc = roc_auc_score(y_dir_true, y_dir_proba)
            except Exception:
                clf_roc_auc = np.nan
            try:
                clf_pr_auc = average_precision_score(y_dir_true, y_dir_proba)
            except Exception:
                clf_pr_auc = np.nan
            try:
                clf_brier = brier_score_loss(y_dir_true, y_dir_proba)
            except Exception:
                clf_brier = np.nan
            try:
                clf_logloss = log_loss(y_dir_true, y_dir_proba)
            except Exception:
                clf_logloss = np.nan

            y_dir_from_reg = (y_reg_pred > 0).astype(int)
            dir_hit_from_reg = accuracy_score(y_dir_true, y_dir_from_reg)

            fold_acc_scores.append(clf_accuracy)

            results_list.append(
                {
                    "commodity": commodity,
                    "window_length": L,
                    "fold_index": fold_index,
                    "test_week": pd.to_datetime(test_week).strftime("%Y-%m-%d"),
                    "n_train": int(n_train_total),
                    "n_test": int(X_test.shape[0]),
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
                    "avg_inference_time": avg_inference_time,
                }
            )

            test_indices = np.where(test_mask)[0]
            for j, idx in enumerate(test_indices):
                row = commodity_df.iloc[idx]
                predictions_list.append(
                    {
                        "commodity": commodity,
                        "window_length": L,
                        "Date": row["Date"].strftime("%Y-%m-%d"),
                        "week": row["week"].strftime("%Y-%m-%d"),
                        "return_next_true": float(y_reg_true[j]),
                        "return_next_pred_lstm": float(y_reg_pred[j]),
                        "direction_true": int(y_dir_true[j]),
                        "direction_pred_lstm": int(y_dir_pred[j]),
                        "direction_pred_proba_lstm": float(y_dir_proba[j]),
                    }
                )

            artifact = {
                "commodity": commodity,
                "window_length": L,
                "fold_index": fold_index,
                "test_week": pd.to_datetime(test_week),
                "input_size": input_size,
                "hidden_size": HIDDEN_SIZE,
                "dropout": DROPOUT,
                "feature_names": feature_columns,
            }

            artifact_path = OUTPUT_DIR / f"{commodity}_L{L}_fold{fold_index}_lstm.pt"
            torch.save(model.state_dict(), artifact_path)

            scaler_path = OUTPUT_DIR / f"{commodity}_L{L}_fold{fold_index}_scaler.joblib"
            joblib.dump(scaler, scaler_path)

            artifact_index.append(
                {
                    "commodity": commodity,
                    "window_length": L,
                    "fold_index": fold_index,
                    "test_week": pd.to_datetime(test_week).strftime("%Y-%m-%d"),
                    "model_path": str(artifact_path),
                    "scaler_path": str(scaler_path),
                    "input_size": input_size,
                    "hidden_size": HIDDEN_SIZE,
                    "dropout": DROPOUT,
                }
            )

        if fold_r2_scores:
            r2_var = float(np.var(fold_r2_scores))
            print(f"\n📈 {commodity} (L={L}) - R² variance across folds: {r2_var:.6f}")
        if fold_acc_scores:
            acc_var = float(np.var(fold_acc_scores))
            print(f"📈 {commodity} (L={L}) - Accuracy variance across folds: {acc_var:.6f}")

# =============================================================================
# 4. Save results and summaries
# =============================================================================

print("\n💾 Saving LSTM results and artifacts...")

results_df = pd.DataFrame(results_list)
predictions_df = pd.DataFrame(predictions_list)
artifact_index_df = pd.DataFrame(artifact_index)

results_path = OUTPUT_DIR / "lstm_fold_results.csv"
predictions_path = OUTPUT_DIR / "lstm_predictions.csv"
artifact_index_path = OUTPUT_DIR / "lstm_artifact_index.csv"
config_path = OUTPUT_DIR / "lstm_config.json"

results_df.to_csv(results_path, index=False)
predictions_df.to_csv(predictions_path, index=False)
artifact_index_df.to_csv(artifact_index_path, index=False)

config = {
    "FAST_MODE": FAST_MODE,
    "TRAIN_START": TRAIN_START,
    "FIRST_TEST": FIRST_TEST,
    "COMMODITIES": COMMODITIES,
    "WINDOW_LENGTHS": WINDOW_LENGTHS,
    "HIDDEN_SIZE": HIDDEN_SIZE,
    "DROPOUT": DROPOUT,
    "BATCH_SIZE": BATCH_SIZE,
    "MAX_EPOCHS": MAX_EPOCHS,
    "EARLY_STOP_PATIENCE": EARLY_STOP_PATIENCE,
    "LEARNING_RATE": LEARNING_RATE,
    "LAMBDA_DIR": LAMBDA_DIR,
    "MAX_FOLDS": MAX_FOLDS,
    "device": str(DEVICE),
    "feature_columns": feature_columns,
}

with open(config_path, "w") as f:
    json.dump(config, f, indent=2)

print(f"  ✓ Fold results       -> {results_path}")
print(f"  ✓ Predictions        -> {predictions_path}")
print(f"  ✓ Artifact index     -> {artifact_index_path}")
print(f"  ✓ Config             -> {config_path}")

if not results_df.empty:
    print("\n📊 OVERALL LSTM PERFORMANCE (averaged):")
    print(
        results_df.groupby("commodity")[["reg_mae", "reg_rmse", "reg_r2", "clf_accuracy"]].mean()
    )
else:
    print("\n⚠️  No results were produced. Check data coverage and FIRST_TEST configuration.")

print("\n✅ LSTM training complete.")
print("=" * 80)