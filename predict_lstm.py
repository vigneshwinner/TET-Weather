"""
LSTM Inference Wrapper

- Loads the most recent LSTM model + scaler per commodity and window length
- Rebuilds the latest sliding window from weekly features
- Outputs next-week return and direction predictions
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn

from train_lstm_model import (
    LSTMModel,
    build_windows_for_commodity,
    feature_columns,   # imported from training script
    DEVICE,
)

OUTPUT_DIR = Path("cleaned_data/lstm_artifacts")

def load_config():
    config_path = OUTPUT_DIR / "lstm_config.json"
    with open(config_path, "r") as f:
        return json.load(f)


def load_artifact_index():
    idx_path = OUTPUT_DIR / "lstm_artifact_index.csv"
    return pd.read_csv(idx_path)


def prepare_latest_window(commodity_df, window_length, scaler):
    """
    Build the latest window for a given commodity/length, then scale it.
    Returns X_latest_scaled (1, L, F) or None if not available.
    """
    X_all, y_reg_all, y_dir_all, last_weeks_all = build_windows_for_commodity(
        commodity_df, window_length
    )
    if X_all.shape[0] == 0:
        return None

    # Take the last sequence as "latest"
    X_latest = X_all[-1:]  # (1, L, F)

    # scale with provided scaler
    N, L, F = X_latest.shape
    X_flat = X_latest.reshape(N * L, F)
    X_scaled_flat = scaler.transform(X_flat)
    X_scaled = X_scaled_flat.reshape(N, L, F).astype(np.float32)
    return X_scaled


def main():
    print("=" * 80)
    print("LSTM INFERENCE")
    print("=" * 80)

    config = load_config()
    artifact_index = load_artifact_index()
    commodities = config["COMMODITIES"]
    window_lengths = config["WINDOW_LENGTHS"]

    # Rebuild merged_df in a lighter way
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

        returns_dfs.append(df[["Date", "week", "commodity", "Close", "return", "return_next", "direction_next"]])

    returns_df = pd.concat(returns_dfs, ignore_index=True)

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
    merged_df = merged_df.fillna(0)

    print("\n📈 Latest LSTM forecasts:")
    for commodity in commodities:
        commodity_df = merged_df[merged_df["commodity"] == commodity].copy()
        commodity_df = commodity_df.sort_values("Date").reset_index(drop=True)

        print("\n" + "-" * 80)
        print(f"Commodity: {commodity}")

        for L in window_lengths:
            # find the most recent artifact (max fold_index) for this commodity/L
            mask = (artifact_index["commodity"] == commodity) & (
                artifact_index["window_length"] == L
            )
            subset = artifact_index[mask]
            if subset.empty:
                print(f"  L={L}: no trained model found.")
                continue

            latest = subset.sort_values("fold_index").iloc[-1]
            model_path = latest["model_path"]
            scaler_path = latest["scaler_path"]
            hidden_size = int(latest["hidden_size"])
            input_size = int(latest["input_size"])

            scaler = joblib.load(scaler_path)

            X_latest_scaled = prepare_latest_window(commodity_df, L, scaler)
            if X_latest_scaled is None:
                print(f"  L={L}: no valid latest window.")
                continue

            model = LSTMModel(input_size=input_size, hidden_size=hidden_size, dropout=latest["dropout"])
            state_dict = torch.load(model_path, map_location=DEVICE)
            model.load_state_dict(state_dict)
            model.to(DEVICE)
            model.eval()

            with torch.no_grad():
                x_tensor = torch.from_numpy(X_latest_scaled).to(DEVICE)
                pred_reg, pred_logits = model(x_tensor)
                pred_reg = pred_reg.cpu().numpy()[0]
                pred_proba = torch.sigmoid(pred_logits).cpu().numpy()[0]
                pred_dir = int(pred_proba > 0.5)

            print(
                f"  L={L}: predicted next-week return={pred_reg:.5f}, "
                f"direction={pred_dir} (p={pred_proba:.3f})"
            )

    print("\n✅ LSTM inference complete.")
    print("=" * 80)


if __name__ == "__main__":
    main()
