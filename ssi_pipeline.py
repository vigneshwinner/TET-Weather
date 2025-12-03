#!/usr/bin/env python3
"""
Supply-Stress Index (SSI) Pipeline
===================================
Constructs a composite Supply-Stress Index combining weather and supply pressure.

Features combined:
- temp_z: Standardized temperature deviation
- precip_z: Standardized precipitation deviation  
- inv_delta_norm: Normalized inventory change
- prod_delta_norm: Normalized production change
- utilization_norm: Normalized utilization rate

Returns: ssi_df (DataFrame with commodity, week_start, ssi_value), loadings (DataFrame)
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
    """Compute temperature and precipitation anomalies (z-scores) from seasonal norms."""
    
    df = weekly_weather.copy()
    df['week_of_year'] = df['week_start'].dt.isocalendar().week
    
    seasonal_stats = df.groupby('week_of_year').agg({
        'temp_avg_c': ['mean', 'std'],
        'precipitation_mm': ['mean', 'std']
    })
    seasonal_stats.columns = ['temp_mean', 'temp_std', 'precip_mean', 'precip_std']
    seasonal_stats = seasonal_stats.reset_index()
    
    df = df.merge(seasonal_stats, on='week_of_year', how='left')
    
    df['temp_z'] = (df['temp_avg_c'] - df['temp_mean']) / df['temp_std'].replace(0, 1)
    df['precip_z'] = (df['precipitation_mm'] - df['precip_mean']) / df['precip_std'].replace(0, 1)
    
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
    return supply_df.merge(
        weather_df[['week_start', 'temp_z', 'precip_z']],
        on='week_start',
        how='inner'
    )


# =============================================================================
# SSI COMPUTATION
# =============================================================================

def compute_ssi(df: pd.DataFrame, variance_threshold: float = 0.85) -> tuple:
    """
    Compute Supply-Stress Index using PCA.
    
    Returns: (df_with_ssi, loadings, explained_variance, n_components)
    """
    
    feature_cols = ['temp_z', 'precip_z', 'inv_delta_norm', 'prod_delta_norm', 'utilization_norm']
    
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
    
    df[feature_cols] = df[feature_cols].fillna(0)
    X = df[feature_cols].values
    
    pca_full = PCA()
    pca_full.fit(X)
    
    cumulative_var = np.cumsum(pca_full.explained_variance_ratio_)
    n_components = np.argmax(cumulative_var >= variance_threshold) + 1
    n_components = max(1, min(n_components, len(feature_cols)))
    
    print(f"\n{'='*60}")
    print("PCA ANALYSIS")
    print(f"{'='*60}")
    print(f"\nVariance explained:")
    for i, var in enumerate(pca_full.explained_variance_ratio_):
        marker = "←" if i < n_components else ""
        print(f"  PC{i+1}: {var*100:.2f}% (cumulative: {cumulative_var[i]*100:.2f}%) {marker}")
    print(f"\nRetaining {n_components} component(s) explaining {cumulative_var[n_components-1]*100:.2f}% variance")
    
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(X)
    pc1 = principal_components[:, 0]
    
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=[f'PC{i+1}' for i in range(n_components)],
        index=feature_cols
    )
    
    stress_direction = {
        'temp_z': 1,
        'precip_z': -1,
        'inv_delta_norm': -1,
        'prod_delta_norm': -1,
        'utilization_norm': 1
    }
    
    direction_score = sum(loadings.loc[feat, 'PC1'] * stress_direction.get(feat, 0) for feat in feature_cols)
    sign_adj = 1 if direction_score >= 0 else -1
    
    df = df.copy()
    df['ssi_value'] = pc1 * sign_adj
    loadings['PC1'] = loadings['PC1'] * sign_adj
    
    print(f"\nSign adjustment: {sign_adj} (positive SSI = higher stress)")
    
    return df, loadings, pca_full.explained_variance_ratio_, n_components


# =============================================================================
# VISUALIZATION (displays inline, no saving)
# =============================================================================

def plot_component_loadings(loadings: pd.DataFrame):
    """Bar chart of PC1 loadings."""
    plt.figure(figsize=(10, 6))
    
    colors = ['#e74c3c' if x > 0 else '#3498db' for x in loadings['PC1']]
    bars = plt.barh(loadings.index, loadings['PC1'], color=colors, edgecolor='black')
    
    plt.axvline(x=0, color='black', linewidth=0.8)
    plt.xlabel('Loading on PC1 (Sign-Adjusted)', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    plt.title('SSI Component Loadings\n(Red = Increases Stress, Blue = Decreases Stress)', fontsize=13)
    
    for bar, val in zip(bars, loadings['PC1']):
        plt.text(val + 0.02 if val >= 0 else val - 0.02,
                 bar.get_y() + bar.get_height()/2,
                 f'{val:.3f}', ha='left' if val >= 0 else 'right', va='center', fontsize=10)
    
    plt.tight_layout()
    plt.show()


def plot_variance_explained(explained_var: np.ndarray, n_retained: int):
    """Variance explained bar chart."""
    plt.figure(figsize=(10, 5))
    
    n_comp = len(explained_var)
    x = np.arange(1, n_comp + 1)
    colors = ['#27ae60' if i < n_retained else '#bdc3c7' for i in range(n_comp)]
    
    bars = plt.bar(x, explained_var * 100, color=colors, edgecolor='black')
    
    cumulative = np.cumsum(explained_var) * 100
    plt.plot(x, cumulative, 'ro-', linewidth=2, markersize=8, label='Cumulative')
    plt.axhline(y=85, color='red', linestyle='--', alpha=0.7, label='85% Threshold')
    
    plt.xlabel('Principal Component', fontsize=12)
    plt.ylabel('Variance Explained (%)', fontsize=12)
    plt.title('PCA Variance Explained (Green = Retained)', fontsize=13)
    plt.xticks(x)
    plt.legend(loc='right')
    
    for bar, val in zip(bars, explained_var * 100):
        plt.text(bar.get_x() + bar.get_width()/2, val + 1, f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
    
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
    ax.set_xticklabels([pivot.columns[i].strftime('%Y-%m') for i in range(0, n_dates, step)], rotation=45, ha='right')
    
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
    Main pipeline to construct Supply-Stress Index.
    
    Args:
        weather_years: Years of weather data to fetch
        show_plots: Whether to display visualizations
    
    Returns:
        ssi_df: DataFrame with columns [commodity, week_start, ssi_value]
        loadings: DataFrame with PCA component loadings
    """
    
    print("="*60)
    print("SUPPLY-STRESS INDEX (SSI) PIPELINE")
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
    print("\n[4/5] Merging and computing SSI...")
    merged_df = merge_weather_and_supply(weather_anomalies, commodity_df)
    print(f"  Merged records: {len(merged_df)}")
    print(f"  Date range: {merged_df['week_start'].min().date()} to {merged_df['week_start'].max().date()}")
    
    ssi_df, loadings, explained_var, n_components = compute_ssi(merged_df, VARIANCE_THRESHOLD)
    
    # =========================================================================
    # STEP 4: Visualize
    # =========================================================================
    if show_plots:
        print("\n[5/5] Generating visualizations...")
        plot_component_loadings(loadings)
        plot_variance_explained(explained_var, n_components)
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
    print("SSI PIPELINE COMPLETE!")
    print("="*60)
    
    # Return clean output DataFrame
    output_df = ssi_df[['commodity', 'week_start', 'ssi_value']].copy()
    output_df = output_df.sort_values(['commodity', 'week_start']).reset_index(drop=True)
    
    return output_df, loadings


if __name__ == "__main__":
    ssi_df, loadings = run_ssi_pipeline()
    
    print("\n\nReturned DataFrames:")
    print("\nssi_df.head():")
    print(ssi_df.head(10))
    print("\nloadings:")
    print(loadings)
