#!/usr/bin/env python3
"""
EIADataFetcher

Thin wrapper around the EIA v2 /seriesid/ endpoint that:
- Pulls weekly fundamentals for crude, nat gas, gasoline, distillate
- Returns a merged pandas DataFrame
- Adds basic standardization + WoW features

This is designed to be used by eia_data_processor.run_complete_pipeline()
"""

import logging
from typing import Dict, Tuple, Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )


class EIADataFetcher:
    """
    Fetches and standardizes weekly commodity data from EIA.
    """

    def __init__(self, api_key: str):
        self.api_key = api_key

        # Map from our internal column name -> (EIA v1-style series ID, units)
        # We send these IDs to the v2 /seriesid/ translator.
        self.series_map: Dict[str, Tuple[str, str]] = {
            # Crude oil
            "crude_oil_inventory": ("PET.WCESTUS1.W", "Thousand Barrels"),
            "crude_oil_production": ("PET.WCRFPUS2.W", "Thousand Barrels per Day"),
            "crude_oil_imports": ("PET.WCEIMUS2.W", "Thousand Barrels per Day"),
            "refinery_utilization": ("PET.WPULEUS3.W", "Percent"),

            # Natural gas
            # Treat storage as "inventory" for signal/feature purposes
            "natural_gas_inventory": ("NG.NW2_EPG0_SWO_R48_BCF.W", "Billion Cubic Feet"),

            # Gasoline
            "gasoline_inventory": ("PET.WGTSTUS1.W", "Thousand Barrels"),
            "gasoline_production": ("PET.WGFUPUS2.W", "Thousand Barrels per Day"),

            # Distillate
            "distillate_inventory": ("PET.WDISTUS1.W", "Thousand Barrels"),
        }

        # v2 translator endpoint – we append the series ID
        self.base_url = "https://api.eia.gov/v2/seriesid/"

    # ------------------------------------------------------------------ #
    # Low-level fetch
    # ------------------------------------------------------------------ #
    def _fetch_single_series(self, series_id: str) -> pd.DataFrame:
        """
        Hit /v2/seriesid/<series_id> and return a small DataFrame with
        columns ["date", "value"].
        """
        url = f"{self.base_url}{series_id}"
        params = {"api_key": self.api_key}

        resp = requests.get(url, params=params, timeout=30)

        if resp.status_code == 403:
            raise RuntimeError("Invalid or unauthorized EIA API key (HTTP 403)")
        if resp.status_code == 404:
            raise RuntimeError(f"Series ID {series_id} not found on EIA API (HTTP 404)")
        resp.raise_for_status()

        payload = resp.json()
        response = payload.get("response", {})
        data_list = response.get("data", [])

        if not data_list:
            raise RuntimeError(f"No data returned for series {series_id}")

        df = pd.DataFrame(data_list)

        if "period" not in df.columns or "value" not in df.columns:
            raise RuntimeError(
                f"Unexpected schema for {series_id}: columns={list(df.columns)}"
            )

        # Normalize columns
        df = df[["period", "value"]].rename(columns={"period": "date"})
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])
        df["value"] = pd.to_numeric(df["value"], errors="coerce")

        return df

    # ------------------------------------------------------------------ #
    # Public API used by the processor
    # ------------------------------------------------------------------ #
    def fetch_all_commodities(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fetch all configured series, merge by date, and optionally
        filter by [start_date, end_date].

        Dates are filtered in pandas so we don't depend on API-side filters.
        """
        all_frames = []

        for col_name, (series_id, units) in self.series_map.items():
            logger.info("Fetching %s (%s)", col_name, series_id)
            try:
                s_df = self._fetch_single_series(series_id)
                s_df = s_df.sort_values("date")
                s_df.rename(columns={"value": col_name}, inplace=True)
                all_frames.append(s_df)
                logger.info("  ✓ got %d rows for %s", len(s_df), col_name)
            except Exception as e:
                logger.error("  ✗ error fetching %s: %s", col_name, e)

        if not all_frames:
            logger.error("No series fetched successfully.")
            return pd.DataFrame()

        # Outer-merge all series on date
        merged = all_frames[0]
        for df in all_frames[1:]:
            merged = merged.merge(df, on="date", how="outer")

        merged = merged.sort_values("date")

        # Optional date filtering (in pandas)
        if start_date is not None:
            start_ts = pd.to_datetime(start_date)
            merged = merged[merged["date"] >= start_ts]
        if end_date is not None:
            end_ts = pd.to_datetime(end_date)
            merged = merged[merged["date"] <= end_ts]

        merged.reset_index(drop=True, inplace=True)
        logger.info("Merged data shape after date filter: %s", merged.shape)

        return merged

    def standardize_and_clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Basic cleaning + standard features:
        - Ensure proper dtypes
        - Week-over-week changes and % changes
        - 4-week and 13-week moving averages
        - Calendar & seasonal flags
        """
        if df.empty:
            logger.warning("standardize_and_clean called with empty DataFrame.")
            return df.copy()

        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

        value_cols = [c for c in df.columns if c != "date"]
        for c in value_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        # Week-over-week changes (current minus previous week)
        for c in value_cols:
            df[f"{c}_wow_change"] = df[c].diff()
            df[f"{c}_wow_pct_change"] = df[c].pct_change() * 100

        # Moving averages
        for c in value_cols:
            df[f"{c}_ma4w"] = df[c].rolling(window=4, min_periods=1).mean()
            df[f"{c}_ma13w"] = df[c].rolling(window=13, min_periods=1).mean()

        # Calendar features
        df["year"] = df["date"].dt.year
        df["month"] = df["date"].dt.month
        df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
        df["quarter"] = df["date"].dt.quarter

        # Seasonal flags (used by processor + signals)
        df["is_summer_driving"] = df["month"].isin([5, 6, 7, 8, 9]).astype(int)
        df["is_heating_season"] = df["month"].isin([11, 12, 1, 2, 3]).astype(int)

        logger.info("Standardized data shape: %s", df.shape)
        return df


# Optional quick test if you run this file directly
if __name__ == "__main__":
    import getpass

    key = 'JMlLALgGbXN9BT2khJUocOZzsuJsdGTACakEAEn8'
    fetcher = EIADataFetcher(key)
    data = fetcher.fetch_all_commodities()
    data = fetcher.standardize_and_clean(data)
    print(data.tail())
    output_file = "eia_raw_standardized_data.csv"
    data.to_csv(output_file, index=False)
    print(f"\nSaved data to {output_file}")
