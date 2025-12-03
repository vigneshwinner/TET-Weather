#!/usr/bin/env python3
"""
EIA Deltas Processor
=====================
Derive week-over-week changes and normalized variations from EIA fundamentals.

Computes for each series:
- delta: value_t - value_{t-1}
- pct_change: (delta / value_{t-1}) * 100
- seasonal_norm: value / same-week 5-year average

Returns: DataFrame with all computed metrics
"""

import pandas as pd
import numpy as np
import warnings

# Import from existing project files
from eia_data_processor import EIADataFetcher

warnings.filterwarnings('ignore')

# Configuration
EIA_API_KEY = 'JMlLALgGbXN9BT2khJUocOZzsuJsdGTACakEAEn8'

# Series to process (inventory, production, utilization)
SERIES_TO_PROCESS = [
    'crude_inventory',
    'crude_production',
    'refinery_utilization',
    'natgas_inventory',
    'gasoline_inventory',
    'gasoline_production',
    'distillate_inventory',
    'distillate_production',
    'propane_inventory',
]


def compute_deltas(df: pd.DataFrame, series_cols: list) -> pd.DataFrame:
    """
    Compute week-over-week changes for each series.
    
    For each series:
    - {series}_delta: value_t - value_{t-1}
    - {series}_pct_change: (delta / value_{t-1}) * 100
    
    Args:
        df: DataFrame with date and series columns
        series_cols: List of column names to process
        
    Returns:
        DataFrame with delta and pct_change columns added
    """
    df = df.copy()
    df = df.sort_values('date').reset_index(drop=True)
    
    for col in series_cols:
        if col not in df.columns:
            print(f"  Warning: {col} not found, skipping")
            continue
            
        # Week-over-week delta
        df[f'{col}_delta'] = df[col].diff()
        
        # Percentage change: (delta / previous_value) * 100
        prev_value = df[col].shift(1)
        df[f'{col}_pct_change'] = (df[f'{col}_delta'] / prev_value) * 100
    
    return df


def compute_seasonal_normalization(df: pd.DataFrame, series_cols: list, lookback_years: int = 5) -> pd.DataFrame:
    """
    Normalize values by dividing by same-week average from prior years.
    
    For each series:
    - {series}_seasonal_avg: Average value for this week-of-year over lookback period
    - {series}_seasonal_norm: value / seasonal_avg (ratio, 1.0 = at historical average)
    
    Args:
        df: DataFrame with date and series columns
        series_cols: List of column names to process
        lookback_years: Number of years to use for seasonal average
        
    Returns:
        DataFrame with seasonal normalization columns added
    """
    df = df.copy()
    
    # Extract week of year
    df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)
    df['year'] = df['date'].dt.year
    
    for col in series_cols:
        if col not in df.columns:
            continue
        
        # Compute rolling seasonal average for each row
        seasonal_avgs = []
        seasonal_norms = []
        
        for idx, row in df.iterrows():
            current_week = row['week_of_year']
            current_year = row['year']
            
            # Get same week from prior years (up to lookback_years)
            mask = (
                (df['week_of_year'] == current_week) & 
                (df['year'] < current_year) & 
                (df['year'] >= current_year - lookback_years)
            )
            
            historical_values = df.loc[mask, col]
            
            if len(historical_values) > 0:
                seasonal_avg = historical_values.mean()
            else:
                seasonal_avg = np.nan
            
            seasonal_avgs.append(seasonal_avg)
            
            # Compute normalized value (ratio to seasonal average)
            if pd.notna(seasonal_avg) and seasonal_avg != 0:
                seasonal_norms.append(row[col] / seasonal_avg)
            else:
                seasonal_norms.append(np.nan)
        
        df[f'{col}_seasonal_avg'] = seasonal_avgs
        df[f'{col}_seasonal_norm'] = seasonal_norms
    
    return df


def handle_invalid_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle NaN and infinite values.
    
    - Replace inf/-inf with NaN
    - Report counts of invalid values
    """
    df = df.copy()
    
    # Count infinities before replacement
    inf_counts = {}
    for col in df.select_dtypes(include=[np.number]).columns:
        inf_count = np.isinf(df[col]).sum()
        if inf_count > 0:
            inf_counts[col] = inf_count
    
    if inf_counts:
        print("\n  Infinite values replaced with NaN:")
        for col, count in inf_counts.items():
            print(f"    {col}: {count}")
    
    # Replace infinities with NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Report NaN counts
    nan_counts = df.isnull().sum()
    nan_cols = nan_counts[nan_counts > 0]
    
    if len(nan_cols) > 0:
        print("\n  NaN counts by column:")
        for col, count in nan_cols.items():
            if count > 0 and col not in ['date', 'week_of_year', 'year']:
                pct = count / len(df) * 100
                print(f"    {col}: {count} ({pct:.1f}%)")
    
    return df


def process_eia_deltas(lookback_years: int = 5, save_csv: bool = False, csv_path: str = './eia_deltas.csv') -> pd.DataFrame:
    """
    Main function to process EIA data and compute all deltas/normalizations.
    
    Args:
        lookback_years: Years of history for seasonal normalization
        save_csv: If True, save output to CSV file
        csv_path: Path for CSV output
        
    Returns:
        DataFrame with all computed metrics
    """
    print("="*60)
    print("EIA DELTAS PROCESSOR")
    print("="*60)
    
    # =========================================================================
    # STEP 1: Fetch EIA Data
    # =========================================================================
    print("\n[1/4] Fetching EIA data...")
    
    fetcher = EIADataFetcher(EIA_API_KEY)
    df = fetcher.fetch_all()
    
    # Ensure date is datetime
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    print(f"  Loaded {len(df)} weekly records")
    print(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    
    # Filter to available series
    available_series = [col for col in SERIES_TO_PROCESS if col in df.columns]
    print(f"  Series to process: {available_series}")
    
    # =========================================================================
    # STEP 2: Compute Deltas
    # =========================================================================
    print("\n[2/4] Computing week-over-week changes...")
    
    df = compute_deltas(df, available_series)
    
    delta_cols = [col for col in df.columns if '_delta' in col or '_pct_change' in col]
    print(f"  Added {len(delta_cols)} delta/pct_change columns")
    
    # =========================================================================
    # STEP 3: Seasonal Normalization
    # =========================================================================
    print(f"\n[3/4] Computing seasonal normalization ({lookback_years}-year average)...")
    
    df = compute_seasonal_normalization(df, available_series, lookback_years)
    
    seasonal_cols = [col for col in df.columns if '_seasonal' in col]
    print(f"  Added {len(seasonal_cols)} seasonal columns")
    
    # =========================================================================
    # STEP 4: Handle Invalid Values
    # =========================================================================
    print("\n[4/4] Validating data (NaN/inf handling)...")
    
    df = handle_invalid_values(df)
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "="*60)
    print("PROCESSING COMPLETE")
    print("="*60)
    
    print(f"\nOutput DataFrame: {df.shape[0]} rows × {df.shape[1]} columns")
    
    # Show sample of computed columns
    print("\nSample computed columns:")
    sample_series = available_series[0] if available_series else None
    if sample_series:
        sample_cols = ['date', sample_series, 
                       f'{sample_series}_delta', 
                       f'{sample_series}_pct_change',
                       f'{sample_series}_seasonal_norm']
        sample_cols = [c for c in sample_cols if c in df.columns]
        print(df[sample_cols].tail(5).to_string(index=False))
    
    # Show column groups
    print("\n" + "="*60)
    print("COLUMN SUMMARY")
    print("="*60)
    
    print("\nRaw values:")
    print(f"  {available_series}")
    
    print("\nDelta columns (value_t - value_{t-1}):")
    print(f"  {[f'{s}_delta' for s in available_series]}")
    
    print("\nPercent change columns ((delta/prev)*100):")
    print(f"  {[f'{s}_pct_change' for s in available_series]}")
    
    print("\nSeasonal norm columns (value / 5yr avg):")
    print(f"  {[f'{s}_seasonal_norm' for s in available_series]}")
    
    # Save to CSV if requested
    if save_csv:
        df.to_csv(csv_path, index=False)
        print(f"\n✓ Saved to: {csv_path}")
    
    return df


def get_eia_deltas(lookback_years: int = 5, save_csv: bool = False, csv_path: str = './eia_deltas.csv') -> pd.DataFrame:
    """
    Convenience function to get processed EIA deltas DataFrame.
    
    Args:
        lookback_years: Years of history for seasonal normalization
        save_csv: If True, save output to CSV file
        csv_path: Path for CSV output
        
    Returns:
        DataFrame with columns:
        - date
        - Original series (crude_inventory, etc.)
        - Delta columns ({series}_delta)
        - Percent change columns ({series}_pct_change)
        - Seasonal average columns ({series}_seasonal_avg)
        - Seasonal normalized columns ({series}_seasonal_norm)
    """
    return process_eia_deltas(lookback_years, save_csv, csv_path)


if __name__ == "__main__":
    # Save CSV when running directly
    df = process_eia_deltas(save_csv=True, csv_path='./eia_deltas.csv')
    
    print("\n\nReturned DataFrame:")
    print(f"\nShape: {df.shape}")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nHead:")
    print(df.head())
