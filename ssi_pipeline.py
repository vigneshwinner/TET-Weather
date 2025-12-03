#!/usr/bin/env python3
"""
Supply-Stress Index (SSI) Pipeline v2
======================================
Per-commodity SSI combining weather and supply pressure.

Features combined (per commodity):
- temp_z: Temperature anomaly (z-score)
- precip_z: Precipitation anomaly (z-score)
- inv_delta_norm: Normalized inventory change
- prod_delta_norm: Normalized production change
- utilization_norm: Normalized utilization rate

Returns: ssi_df (DataFrame), loadings_dict (dict of DataFrames per commodity)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings

# Import from existing project files
from eia_data_processor import EIADataFetcher
from get_nasa_data import get_weather_data

warnings.filterwarnings('ignore')

# Configuration
VARIANCE_THRESHOLD = 0.85
EIA_API_KEY = 'JMlLALgGbXN9BT2khJUocOZzsuJsdGTACakEAEn8'

FEATURE_COLS = [
    'temp_z', 'precip_z',
    'inv_delta_norm', 'prod_delta_norm', 'utilization_norm'
]


# =============================================================================
# DATA PROCESSING FUNCTIONS
# =============================================================================

def aggregate_weather_weekly(weather_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate daily weather data to weekly, computing regional averages."""
    
    if weather_df.empty:
        return pd.DataFrame()
    
    weather_df = weather_df.copy()
    weather_df['week_start'] = weather_df['date'].dt.to_period('W-SUN').dt.start_time
    
    weekly = weather_df.groupby('week_start').agg({
        'temp_avg_c': 'mean',
        'precipitation_mm': 'sum',
    }).reset_index()
    
    return weekly


def compute_weather_anomalies(weekly_weather: pd.DataFrame) -> pd.DataFrame:
    """Compute temperature and precipitation anomalies (z-scores)."""
    
    df = weekly_weather.copy()
    df = df.sort_values('week_start').reset_index(drop=True)
    df['week_of_year'] = df['week_start'].dt.isocalendar().week
    
    seasonal_stats = df.groupby('week_of_year').agg({
        'temp_avg_c': ['mean', 'std'],
        'precipitation_mm': ['mean', 'std']
    })
    seasonal_stats.columns = ['temp_mean', 'temp_std', 'precip_mean', 'precip_std']
    seasonal_stats = seasonal_stats.reset_index()
    
    df = df.merge(seasonal_stats, on='week_of_year', how='left')
    
    # Compute z-scores
    df['temp_z'] = (df['temp_avg_c'] - df['temp_mean']) / df['temp_std'].replace(0, 1)
    df['precip_z'] = (df['precipitation_mm'] - df['precip_mean']) / df['precip_std'].replace(0, 1)
    
    # Handle infinities
    df['temp_z'] = df['temp_z'].replace([np.inf, -np.inf], np.nan).fillna(0)
    df['precip_z'] = df['precip_z'].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    return df[['week_start', 'temp_avg_c', 'precipitation_mm', 'temp_z', 'precip_z']]


def process_commodity_data(eia_df: pd.DataFrame) -> pd.DataFrame:
    """Process EIA data into commodity-specific records with normalized features."""
    
    df = eia_df.copy()
    df['week_start'] = pd.to_datetime(df['date']).dt.to_period('W-SUN').dt.start_time
    
    commodity_config = {
        'Crude Oil': {
            'inventory': 'crude_inventory',
            'production': 'crude_production',
            'utilization': 'refinery_utilization'
        },
        'Natural Gas': {
            'inventory': 'natgas_inventory',
            'production': 'natgas_net_change',
            'utilization': None
        },
        'Gasoline': {
            'inventory': 'gasoline_inventory',
            'production': 'gasoline_production',
            'utilization': 'refinery_utilization'
        },
        'Distillate': {
            'inventory': 'distillate_inventory',
            'production': 'distillate_production',
            'utilization': 'refinery_utilization'
        },
        'Propane': {
            'inventory': 'propane_inventory',
            'production': None,
            'utilization': None
        }
    }
    
    all_commodities = []
    
    for commodity, config in commodity_config.items():
        comm_df = df[['week_start']].copy()
        comm_df['commodity'] = commodity
        
        if config['inventory'] and config['inventory'] in df.columns:
            comm_df['inventory'] = df[config['inventory']]
            comm_df['inv_delta'] = df[config['inventory']].diff()
        else:
            comm_df['inventory'] = np.nan
            comm_df['inv_delta'] = np.nan
        
        if config['production'] and config['production'] in df.columns:
            comm_df['production'] = df[config['production']]
            comm_df['prod_delta'] = df[config['production']].diff()
        else:
            comm_df['production'] = np.nan
            comm_df['prod_delta'] = np.nan
        
        if config['utilization'] and config['utilization'] in df.columns:
            comm_df['utilization'] = df[config['utilization']]
        else:
            comm_df['utilization'] = np.nan
        
        all_commodities.append(comm_df)
    
    result = pd.concat(all_commodities, ignore_index=True)
    
    # Normalize features within each commodity
    for commodity in result['commodity'].unique():
        mask = result['commodity'] == commodity
        
        for col in ['inv_delta', 'prod_delta', 'utilization']:
            if result.loc[mask, col].notna().sum() > 1:
                scaler = StandardScaler()
                values = result.loc[mask, col].values.reshape(-1, 1)
                valid_mask = ~np.isnan(values.flatten())
                if valid_mask.sum() > 1:
                    scaler.fit(values[valid_mask].reshape(-1, 1))
                    result.loc[mask, f'{col}_norm'] = scaler.transform(values).flatten()
                else:
                    result.loc[mask, f'{col}_norm'] = 0
            else:
                result.loc[mask, f'{col}_norm'] = 0
    
    return result


def merge_weather_and_supply(weather_df: pd.DataFrame, supply_df: pd.DataFrame) -> pd.DataFrame:
    """Merge weather anomalies with supply data."""
    weather_cols = ['week_start', 'temp_z', 'precip_z']
    return supply_df.merge(weather_df[weather_cols], on='week_start', how='inner')


# =============================================================================
# PER-COMMODITY SSI COMPUTATION
# =============================================================================

def compute_ssi_per_commodity(df: pd.DataFrame, variance_threshold: float = 0.85) -> tuple:
    """
    Compute Supply-Stress Index separately for each commodity.
    
    Returns: (df_with_ssi, loadings_dict, variance_dict)
    """
    
    commodities = df['commodity'].unique()
    all_results = []
    loadings_dict = {}
    variance_dict = {}
    
    print(f"\n{'='*60}")
    print("PER-COMMODITY PCA ANALYSIS")
    print(f"{'='*60}")
    
    for commodity in commodities:
        print(f"\n--- {commodity} ---")
        
        comm_df = df[df['commodity'] == commodity].copy()
        comm_df = comm_df.sort_values('week_start').reset_index(drop=True)
        
        # Determine which features are available for this commodity
        available_features = []
        for col in FEATURE_COLS:
            if col in comm_df.columns:
                # Check if column has meaningful variance
                if comm_df[col].notna().sum() > 1 and comm_df[col].std() > 0.01:
                    available_features.append(col)
                    comm_df[col] = comm_df[col].fillna(0)
                else:
                    comm_df[col] = 0
            else:
                comm_df[col] = 0
        
        # Need at least 2 features for PCA
        if len(available_features) < 2:
            print(f"  Skipping: insufficient features ({available_features})")
            comm_df['ssi_value'] = 0
            all_results.append(comm_df)
            continue
        
        print(f"  Features used: {available_features}")
        
        # Drop rows with NaN in any feature (from lagging)
        comm_df = comm_df.dropna(subset=available_features)
        
        if len(comm_df) < 10:
            print(f"  Skipping: insufficient data ({len(comm_df)} rows)")
            comm_df['ssi_value'] = 0
            all_results.append(comm_df)
            continue
        
        X = comm_df[available_features].values
        
        # Fit PCA
        pca_full = PCA()
        pca_full.fit(X)
        
        cumulative_var = np.cumsum(pca_full.explained_variance_ratio_)
        n_components = np.argmax(cumulative_var >= variance_threshold) + 1
        n_components = max(1, min(n_components, len(available_features)))
        
        print(f"  Variance: PC1={pca_full.explained_variance_ratio_[0]*100:.1f}%, "
              f"cumulative({n_components})={cumulative_var[n_components-1]*100:.1f}%")
        
        pca = PCA(n_components=n_components)
        principal_components = pca.fit_transform(X)
        pc1 = principal_components[:, 0]
        
        # Create loadings DataFrame
        loadings = pd.DataFrame(
            pca.components_.T,
            columns=[f'PC{i+1}' for i in range(n_components)],
            index=available_features
        )
        
        # Sign adjustment based on stress interpretation
        stress_direction = {
            'temp_z': 1,           # Higher temp = stress
            'precip_z': -1,        # Lower precip = stress (drought)
            'inv_delta_norm': -1,  # Declining inventory = stress
            'prod_delta_norm': -1, # Declining production = stress
            'utilization_norm': 1  # Higher utilization = stress
        }
        
        direction_score = sum(
            loadings.loc[feat, 'PC1'] * stress_direction.get(feat, 0) 
            for feat in available_features if feat in stress_direction
        )
        sign_adj = 1 if direction_score >= 0 else -1
        
        comm_df['ssi_value'] = pc1 * sign_adj
        loadings['PC1'] = loadings['PC1'] * sign_adj
        
        print(f"  Sign adjustment: {sign_adj}")
        
        # Store results
        all_results.append(comm_df)
        loadings_dict[commodity] = loadings
        variance_dict[commodity] = pca_full.explained_variance_ratio_
    
    # Combine all commodities
    result_df = pd.concat(all_results, ignore_index=True)
    
    return result_df, loadings_dict, variance_dict


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_loadings_comparison(loadings_dict: dict):
    """Bar chart comparing PC1 loadings across commodities."""
    
    # Get all unique features across commodities
    all_features = set()
    for loadings in loadings_dict.values():
        all_features.update(loadings.index.tolist())
    all_features = sorted(all_features)
    
    n_commodities = len(loadings_dict)
    fig, axes = plt.subplots(1, n_commodities, figsize=(4 * n_commodities, 6), sharey=True)
    
    if n_commodities == 1:
        axes = [axes]
    
    for ax, (commodity, loadings) in zip(axes, loadings_dict.items()):
        features = loadings.index.tolist()
        values = loadings['PC1'].values
        
        colors = ['#e74c3c' if v > 0 else '#3498db' for v in values]
        bars = ax.barh(features, values, color=colors, edgecolor='black')
        
        ax.axvline(x=0, color='black', linewidth=0.8)
        ax.set_xlabel('PC1 Loading')
        ax.set_title(f'{commodity}', fontsize=11, fontweight='bold')
        
        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(val + 0.02 if val >= 0 else val - 0.02,
                   bar.get_y() + bar.get_height()/2,
                   f'{val:.2f}', ha='left' if val >= 0 else 'right', 
                   va='center', fontsize=8)
    
    axes[0].set_ylabel('Features')
    fig.suptitle('SSI Component Loadings by Commodity\n(Red = Increases Stress)', 
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_ssi_timeseries(df: pd.DataFrame):
    """Time series plot by commodity."""
    commodities = df['commodity'].unique()
    n_comm = len(commodities)
    
    fig, axes = plt.subplots(n_comm, 1, figsize=(14, 3 * n_comm), sharex=True)
    if n_comm == 1:
        axes = [axes]
    
    colors = plt.cm.Set2(np.linspace(0, 1, n_comm))
    
    for ax, commodity, color in zip(axes, commodities, colors):
        comm_data = df[df['commodity'] == commodity].sort_values('week_start')
        
        ax.plot(comm_data['week_start'], comm_data['ssi_value'], color=color, linewidth=1.5)
        ax.fill_between(comm_data['week_start'], 0, comm_data['ssi_value'],
                        where=comm_data['ssi_value'] > 0, color='#e74c3c', alpha=0.3, label='High Stress')
        ax.fill_between(comm_data['week_start'], 0, comm_data['ssi_value'],
                        where=comm_data['ssi_value'] <= 0, color='#27ae60', alpha=0.3, label='Low Stress')
        
        ax.axhline(y=0, color='black', linewidth=0.8)
        ax.set_ylabel('SSI', fontsize=10)
        ax.set_title(f'{commodity}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=8)
    
    axes[-1].set_xlabel('Week', fontsize=12)
    fig.suptitle('Supply-Stress Index (SSI) by Commodity', fontsize=14, fontweight='bold', y=1.01)
    
    plt.tight_layout()
    plt.show()


def plot_ssi_heatmap(df: pd.DataFrame):
    """Heatmap of SSI values."""
    pivot = df.pivot_table(index='commodity', columns='week_start', values='ssi_value', aggfunc='mean')
    
    fig, ax = plt.subplots(figsize=(16, 6))
    im = ax.imshow(pivot.values, aspect='auto', cmap='RdYlGn_r')
    
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    
    n_dates = len(pivot.columns)
    step = max(1, n_dates // 12)
    ax.set_xticks(np.arange(0, n_dates, step))
    ax.set_xticklabels([pivot.columns[i].strftime('%Y-%m') for i in range(0, n_dates, step)], 
                       rotation=45, ha='right')
    
    plt.colorbar(im, label='SSI Value', ax=ax)
    ax.set_xlabel('Week', fontsize=12)
    ax.set_ylabel('Commodity', fontsize=12)
    ax.set_title('Supply-Stress Index Heatmap (Red = High Stress)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.show()


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_ssi_pipeline(weather_years=2, show_plots=True):
    """
    Main pipeline to construct per-commodity Supply-Stress Index.
    
    Args:
        weather_years: Years of weather data to fetch
        show_plots: Whether to display visualizations
    
    Returns:
        ssi_df: DataFrame with columns [commodity, week_start, ssi_value]
        loadings_dict: Dict of DataFrames with PCA loadings per commodity
    """
    
    print("="*60)
    print("SUPPLY-STRESS INDEX (SSI) PIPELINE v2")
    print("Per-Commodity Analysis")
    print("="*60)
    
    # =========================================================================
    # STEP 1: Fetch Data
    # =========================================================================
    print("\n[1/5] Fetching data...")
    
    print("  Fetching EIA supply data...")
    eia_fetcher = EIADataFetcher(EIA_API_KEY)
    eia_df = eia_fetcher.fetch_all()
    print(f"  EIA: {len(eia_df)} weeks, {eia_df['date'].min().date()} to {eia_df['date'].max().date()}")
    
    print(f"  Fetching NASA POWER weather data ({weather_years} years)...")
    weather_df = get_weather_data(years=weather_years, return_format='dataframe')
    print(f"  Weather: {len(weather_df)} records")
    
    # =========================================================================
    # STEP 2: Process Data
    # =========================================================================
    print("\n[2/5] Processing weather data...")
    weekly_weather = aggregate_weather_weekly(weather_df)
    weather_anomalies = compute_weather_anomalies(weekly_weather)
    print(f"  Weekly records: {len(weather_anomalies)}")
    
    print("\n[3/5] Processing supply data...")
    commodity_df = process_commodity_data(eia_df)
    print(f"  Commodities: {commodity_df['commodity'].unique().tolist()}")
    
    # =========================================================================
    # STEP 3: Merge & Compute SSI
    # =========================================================================
    print("\n[4/5] Merging and computing per-commodity SSI...")
    merged_df = merge_weather_and_supply(weather_anomalies, commodity_df)
    print(f"  Merged records: {len(merged_df)}")
    print(f"  Date range: {merged_df['week_start'].min().date()} to {merged_df['week_start'].max().date()}")
    
    ssi_df, loadings_dict, variance_dict = compute_ssi_per_commodity(merged_df, VARIANCE_THRESHOLD)
    
    # =========================================================================
    # STEP 4: Visualize
    # =========================================================================
    if show_plots:
        print("\n[5/5] Generating visualizations...")
        plot_loadings_comparison(loadings_dict)
        plot_ssi_timeseries(ssi_df)
        plot_ssi_heatmap(ssi_df)
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "="*60)
    print("SSI SUMMARY BY COMMODITY")
    print("="*60)
    summary = ssi_df.groupby('commodity')['ssi_value'].agg(['mean', 'std', 'min', 'max']).round(3)
    print(summary)
    
    print("\n" + "="*60)
    print("WEATHER FEATURE IMPORTANCE (PC1 Loadings)")
    print("="*60)
    for commodity, loadings in loadings_dict.items():
        weather_features = [f for f in loadings.index if 'temp' in f or 'precip' in f]
        if weather_features:
            weather_loadings = loadings.loc[weather_features, 'PC1']
            total_weather = np.abs(weather_loadings).sum()
            print(f"\n{commodity}:")
            for feat in weather_features:
                print(f"  {feat}: {loadings.loc[feat, 'PC1']:.3f}")
            print(f"  Total weather impact: {total_weather:.3f}")
    
    print("\n" + "="*60)
    print("SSI PIPELINE COMPLETE!")
    print("="*60)
    
    # Return clean output DataFrame
    output_df = ssi_df[['commodity', 'week_start', 'ssi_value']].copy()
    output_df = output_df.sort_values(['commodity', 'week_start']).reset_index(drop=True)
    
    return output_df, loadings_dict


if __name__ == "__main__":
    ssi_df, loadings_dict = run_ssi_pipeline()
    
    print("\n\nReturned DataFrames:")
    print("\nssi_df.head(10):")
    print(ssi_df.head(10))
    print("\nLoadings per commodity:")
    for commodity, loadings in loadings_dict.items():
        print(f"\n{commodity}:")
        print(loadings)
