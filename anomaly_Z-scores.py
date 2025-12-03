"""
Detect weather anomalies in NASA POWER daily data using region-wise z-scores
and generate anomaly threshold tables (mean ± 3*std).

Creates:
  - Full dataset with anomaly info and thresholds (one row per day)
  - Region-level threshold table (one row per region)
"""

from pathlib import Path
import pandas as pd

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
INPUT_PATH = Path("nasa_power_weather_daily.csv")
OUTPUT_FULL = Path("nasa_power_weather_daily_with_anomalies_z3.csv")
OUTPUT_THRESHOLDS = Path("nasa_power_weather_thresholds_z3.csv")

REGION_COL = "region"
DATE_COL = "date"

Z_THRESHOLD = 3.0

WEATHER_VARS = [
    "temp_avg_c",
    "temp_max_c",
    "temp_min_c",
    "precipitation_mm",
    "solar_radiation_kwh_m2",
    "wind_speed_ms",
    "wind_direction_deg",
]


def main():

    # -------------------------------------------------------------------------
    # 1. Load data
    # -------------------------------------------------------------------------
    df = pd.read_csv(INPUT_PATH)

    if DATE_COL in df.columns:
        df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")

    if REGION_COL not in df.columns:
        raise ValueError(f"Expected a '{REGION_COL}' column in the input CSV.")

    missing = [c for c in WEATHER_VARS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing weather columns: {missing}")

    grouped = df.groupby(REGION_COL)

    anomaly_flag_cols = []

    # -------------------------------------------------------------------------
    # 2. Compute region-level stats & anomaly thresholds
    # -------------------------------------------------------------------------
    for col in WEATHER_VARS:

        mean_col = f"{col}_region_mean"
        std_col = f"{col}_region_std"
        z_col = f"{col}_z"
        flag_col = f"{col}_anomaly_z{int(Z_THRESHOLD)}"
        upper_col = f"{col}_upper_z{int(Z_THRESHOLD)}"
        lower_col = f"{col}_lower_z{int(Z_THRESHOLD)}"

        # region-wise mean and std
        df[mean_col] = grouped[col].transform("mean")
        df[std_col] = grouped[col].transform("std")

        # avoid division by zero (std = 0 → z becomes NaN)
        std_nonzero = df[std_col].replace(0, pd.NA)
        df[z_col] = (df[col] - df[mean_col]) / std_nonzero

        # anomaly flag
        df[flag_col] = df[z_col].abs() >= Z_THRESHOLD
        anomaly_flag_cols.append(flag_col)

        # anomaly thresholds (same for all rows in a region)
        df[upper_col] = df[mean_col] + Z_THRESHOLD * df[std_col]
        df[lower_col] = df[mean_col] - Z_THRESHOLD * df[std_col]

    # -------------------------------------------------------------------------
    # 3. Region-level threshold summary table
    # -------------------------------------------------------------------------
    # Build aggregation dictionary programmatically
    agg_dict = {}
    z_int = int(Z_THRESHOLD)
    for col in WEATHER_VARS:
        agg_dict[f"{col}_region_mean"] = "first"
        agg_dict[f"{col}_region_std"] = "first"
        agg_dict[f"{col}_upper_z{z_int}"] = "first"
        agg_dict[f"{col}_lower_z{z_int}"] = "first"

    threshold_table = (
        df.groupby(REGION_COL)
          .agg(agg_dict)
          .reset_index()
    )

    # -------------------------------------------------------------------------
    # 4. Overall anomaly flag (any variable)
    # -------------------------------------------------------------------------
    df[f"any_anomaly_z{z_int}"] = df[anomaly_flag_cols].any(axis=1)

    # -------------------------------------------------------------------------
    # 5. Save outputs
    # -------------------------------------------------------------------------
    df.to_csv(OUTPUT_FULL, index=False)
    threshold_table.to_csv(OUTPUT_THRESHOLDS, index=False)


if __name__ == "__main__":
    main()
