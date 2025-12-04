"""
Create Interaction Features: Weather Stress × Supply/Demand Metrics
Captures nonlinear relationships between weather anomalies and EIA fundamentals
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler

print("="*70)
print("CREATING INTERACTION FEATURES")
print("="*70)

# ============================================================================
# Load Data
# ============================================================================

print("\n📂 Loading data...")

# Weather anomalies with z-scores
weather_df = pd.read_csv('cleaned_data/nasa_power_weather_daily_with_anomalies_z3.csv')
weather_df['date'] = pd.to_datetime(weather_df['date'])
print(f"✓ Weather data: {len(weather_df)} daily records")

# EIA weekly data
eia_df = pd.read_csv('cleaned_data/eia_3yr_data.csv')
eia_df['date'] = pd.to_datetime(eia_df['date'])
print(f"✓ EIA data: {len(eia_df)} weekly records")
print(f"  Date range: {eia_df['date'].min().date()} to {eia_df['date'].max().date()}")

# ============================================================================
# Region-to-Commodity Mapping
# ============================================================================

region_commodity_map = {
    # Crude Oil
    'North_Sea_Norway': 'Crude_Oil',
    
    # Natural Gas
    'Houston_US': 'Natural_Gas',
    'Cushing_Oklahoma': 'Natural_Gas',
    
    # Power (Texas)
    'Dallas_TX': 'Power',
    'Houston_US_Power': 'Power',
    'San_Antonio_TX': 'Power',
    'Tyler_TX': 'Power',
    'Amarillo_TX': 'Power',
    'Midland_TX': 'Power',
    
    # Copper (South America)
    'Peru': 'Copper',
    'Chile': 'Copper',
    
    # Corn (Midwest)
    'Midwest': 'Corn',
}

# Add commodity column to weather data
weather_df['commodity'] = weather_df['region'].map(region_commodity_map)

# For regions that aren't in the map, use heuristics
weather_df.loc[weather_df['region'].str.contains('TX', na=False), 'commodity'] = 'Power'
weather_df.loc[weather_df['region'] == 'Houston_US', 'commodity'] = 'Natural_Gas'

print(f"\n🗺️  Region-Commodity Mapping:")
commodity_counts = weather_df.groupby('commodity')['region'].unique()
for commodity, regions in commodity_counts.items():
    if commodity is not None:
        print(f"  {commodity}: {', '.join(regions)}")

# ============================================================================
# Calculate EIA Deltas (Week-over-Week Changes)
# ============================================================================

print("\n📊 Calculating EIA deltas...")

eia_df = eia_df.sort_values('date').reset_index(drop=True)

# Calculate deltas for key metrics
eia_df['crude_inv_delta'] = eia_df['crude_inventory'].diff()
eia_df['crude_prod_delta'] = eia_df['crude_production'].diff()
eia_df['natgas_inv_delta'] = eia_df['natgas_inventory'].diff()
eia_df['gasoline_inv_delta'] = eia_df['gasoline_inventory'].diff()
eia_df['gasoline_prod_delta'] = eia_df['gasoline_production'].diff()
eia_df['distillate_inv_delta'] = eia_df['distillate_inventory'].diff()
eia_df['util_delta'] = eia_df['refinery_utilization'].diff()

print(f"✓ Created {7} delta features")

# ============================================================================
# Merge Weather with EIA (Daily weather + Weekly EIA)
# ============================================================================

print("\n🔗 Merging weather anomalies with EIA data...")

# Create week column for both datasets (week starting Monday)
weather_df['week'] = weather_df['date'] - pd.to_timedelta(weather_df['date'].dt.dayofweek.astype('int64'), unit='D')  # type: ignore[attr-defined]
eia_df['week'] = eia_df['date'] - pd.to_timedelta(eia_df['date'].dt.dayofweek.astype('int64'), unit='D')  # type: ignore[attr-defined]

# Select key z-score columns from weather
weather_cols = [
    'date', 'week', 'region', 'commodity',
    'temp_avg_c_z', 'temp_max_c_z', 'temp_min_c_z',
    'precipitation_mm_z', 'solar_radiation_kwh_m2_z',
    'wind_speed_ms_z'
]

weather_subset = weather_df[weather_cols].copy()

# Merge on week (daily weather repeats for each week)
merged_df = weather_subset.merge(eia_df, on='week', how='inner', suffixes=('', '_eia'))

print(f"✓ Merged dataset: {len(merged_df)} records")
print(f"  Date range: {merged_df['date'].min().date()} to {merged_df['date'].max().date()}")

# ============================================================================
# Create Interaction Features
# ============================================================================

print("\n⚡ Creating interaction features...")

# Temperature × Inventory Deltas
merged_df['temp_z_x_crude_inv_delta'] = merged_df['temp_avg_c_z'] * merged_df['crude_inv_delta']
merged_df['temp_z_x_natgas_inv_delta'] = merged_df['temp_avg_c_z'] * merged_df['natgas_inv_delta']
merged_df['temp_z_x_gasoline_inv_delta'] = merged_df['temp_avg_c_z'] * merged_df['gasoline_inv_delta']

# Temperature × Production Deltas
merged_df['temp_z_x_crude_prod_delta'] = merged_df['temp_avg_c_z'] * merged_df['crude_prod_delta']
merged_df['temp_z_x_gasoline_prod_delta'] = merged_df['temp_avg_c_z'] * merged_df['gasoline_prod_delta']

# Temperature × Utilization
merged_df['temp_z_x_utilization'] = merged_df['temp_avg_c_z'] * merged_df['refinery_utilization']
merged_df['temp_z_x_util_delta'] = merged_df['temp_avg_c_z'] * merged_df['util_delta']

# Precipitation × Production Deltas
merged_df['precip_z_x_crude_prod_delta'] = merged_df['precipitation_mm_z'] * merged_df['crude_prod_delta']
merged_df['precip_z_x_gasoline_prod_delta'] = merged_df['precipitation_mm_z'] * merged_df['gasoline_prod_delta']

# Precipitation × Inventory Deltas
merged_df['precip_z_x_crude_inv_delta'] = merged_df['precipitation_mm_z'] * merged_df['crude_inv_delta']
merged_df['precip_z_x_natgas_inv_delta'] = merged_df['precipitation_mm_z'] * merged_df['natgas_inv_delta']

# Max Temperature × Critical Metrics (for extreme heat/cold)
merged_df['temp_max_z_x_utilization'] = merged_df['temp_max_c_z'] * merged_df['refinery_utilization']
merged_df['temp_min_z_x_natgas_inv_delta'] = merged_df['temp_min_c_z'] * merged_df['natgas_inv_delta']

# Wind × Production (relevant for power)
merged_df['wind_z_x_crude_prod_delta'] = merged_df['wind_speed_ms_z'] * merged_df['crude_prod_delta']

print(f"✓ Created {14} interaction features")

# ============================================================================
# Create Lagged Interaction Features
# ============================================================================

print("\n⏰ Creating lagged interaction features...")

# Sort by region and date
merged_df = merged_df.sort_values(['region', 'date']).reset_index(drop=True)

# Create lagged weather features (previous week)
lagged_features = []

for region in merged_df['region'].unique():
    region_mask = merged_df['region'] == region
    region_df = merged_df[region_mask].copy()
    
    # Lag weather z-scores by 1 week (7 days)
    region_df['temp_z_lag1'] = region_df['temp_avg_c_z'].shift(7)
    region_df['precip_z_lag1'] = region_df['precipitation_mm_z'].shift(7)
    
    # Lagged interactions (previous week weather × current week delta)
    region_df['temp_z_lag1_x_crude_inv_delta'] = region_df['temp_z_lag1'] * region_df['crude_inv_delta']
    region_df['temp_z_lag1_x_natgas_inv_delta'] = region_df['temp_z_lag1'] * region_df['natgas_inv_delta']
    region_df['precip_z_lag1_x_crude_prod_delta'] = region_df['precip_z_lag1'] * region_df['crude_prod_delta']
    
    lagged_features.append(region_df)

merged_df = pd.concat(lagged_features, ignore_index=True)

print(f"✓ Created {5} lagged interaction features")

# ============================================================================
# Standardize Interaction Features
# ============================================================================

print("\n📏 Standardizing interaction features...")

# Get all interaction columns
interaction_cols = [col for col in merged_df.columns if '_x_' in col]

print(f"  Standardizing {len(interaction_cols)} interaction columns...")

# Standardize (mean=0, std=1)
scaler = StandardScaler()
merged_df[interaction_cols] = scaler.fit_transform(merged_df[interaction_cols].fillna(0))

print(f"✓ Standardized features (mean ≈ 0, std ≈ 1)")

# ============================================================================
# Save Final Dataset
# ============================================================================

print("\n💾 Saving interaction features...")

# Save full dataset
output_file = 'cleaned_data/weather_eia_interactions.csv'
merged_df.to_csv(output_file, index=False)
print(f"✓ Saved: {output_file}")

# ============================================================================
# Summary Statistics
# ============================================================================

print("\n" + "="*70)
print("SUMMARY STATISTICS")
print("="*70)

print(f"\n📊 Dataset Overview:")
print(f"  Total records: {len(merged_df)}")
print(f"  Date range: {merged_df['date'].min().date()} to {merged_df['date'].max().date()}")
print(f"  Regions: {merged_df['region'].nunique()}")
print(f"  Commodities: {merged_df['commodity'].nunique()}")

print(f"\n📊 Feature Summary:")
print(f"  Weather z-scores: 6")
print(f"  EIA metrics: {len([c for c in eia_df.columns if c != 'date' and c != 'week'])}")
print(f"  EIA deltas: 7")
print(f"  Interaction features: {len(interaction_cols)}")
print(f"  Total features: {len(merged_df.columns)}")

print(f"\n📊 Sample Interaction Statistics:")
sample_interactions = [
    'temp_z_x_crude_inv_delta',
    'temp_z_x_natgas_inv_delta',
    'precip_z_x_crude_prod_delta',
    'temp_z_x_utilization'
]

for col in sample_interactions:
    if col in merged_df.columns:
        mean = merged_df[col].mean()
        std = merged_df[col].std()
        min_val = merged_df[col].min()
        max_val = merged_df[col].max()
        print(f"  {col}:")
        print(f"    Mean: {mean:.4f}, Std: {std:.4f}, Range: [{min_val:.4f}, {max_val:.4f}]")

print(f"\n📊 Records by Commodity:")
if 'commodity' in merged_df.columns:
    commodity_counts = merged_df.groupby('commodity').size()
    for commodity, count in commodity_counts.items():
        print(f"  {commodity}: {count:,} records")

print(f"\n📊 Missing Values:")
missing = merged_df[interaction_cols].isnull().sum()
if missing.sum() > 0:
    print(f"  Columns with missing values:")
    for col, count in missing[missing > 0].items():
        print(f"    {col}: {count} ({count/len(merged_df)*100:.1f}%)")
else:
    print(f"  ✓ No missing values in interaction features")

print("\n" + "="*70)
print("✅ INTERACTION FEATURES CREATED!")
print("="*70)
print("\n🎯 Next Steps:")
print("  1. Use weather_eia_interactions.csv for model training")
print("  2. Features are standardized and ready for ML models")
print("  3. Consider feature selection based on correlation with target")
print("  4. Test lagged interactions for predictive power")
print("\n✨ Nonlinear weather-supply relationships captured!")
print("="*70 + "\n")
