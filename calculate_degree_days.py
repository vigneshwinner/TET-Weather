"""
Calculate Heating Degree Days (HDD) and Cooling Degree Days (CDD)
Generate rolling averages and aggregate to weekly sums
"""

import pandas as pd
import numpy as np
from datetime import datetime

print("="*70)
print("CALCULATING DEGREE DAYS (HDD & CDD)")
print("="*70)

# Load NASA POWER weather data
print("\n📂 Loading NASA POWER weather data...")
df = pd.read_csv('cleaned_data/nasa_weather_daily_5_Years.csv')
df['date'] = pd.to_datetime(df['date'])

print(f"✓ Loaded {len(df)} daily records")
print(f"✓ Date range: {df['date'].min().date()} to {df['date'].max().date()}")
print(f"✓ Regions: {len(df['region'].unique())}")
print(f"  {', '.join(df['region'].unique())}")

# Convert Celsius to Fahrenheit
df['temp_avg_f'] = (df['temp_avg_c'] * 9/5) + 32

# Base temperature for degree days (65°F standard)
BASE_TEMP = 65.0

# Calculate HDD and CDD
print(f"\n🌡️  Calculating Degree Days (Base: {BASE_TEMP}°F)...")
df['hdd'] = df['temp_avg_f'].apply(lambda t: max(0, BASE_TEMP - t))
df['cdd'] = df['temp_avg_f'].apply(lambda t: max(0, t - BASE_TEMP))

print(f"✓ HDD range: {df['hdd'].min():.2f} to {df['hdd'].max():.2f}")
print(f"✓ CDD range: {df['cdd'].min():.2f} to {df['cdd'].max():.2f}")

# Sort by region and date
df = df.sort_values(['region', 'date']).reset_index(drop=True)

# Calculate rolling averages (7, 14, 30 days) for each region
print("\n📊 Calculating Rolling Averages...")

degree_days_list = []

for region in df['region'].unique():
    print(f"  Processing {region}...")
    region_df = df[df['region'] == region].copy()
    
    # 7-day rolling averages
    region_df['hdd_7'] = region_df['hdd'].rolling(window=7, min_periods=1).mean()
    region_df['cdd_7'] = region_df['cdd'].rolling(window=7, min_periods=1).mean()
    
    # 14-day rolling averages
    region_df['hdd_14'] = region_df['hdd'].rolling(window=14, min_periods=1).mean()
    region_df['cdd_14'] = region_df['cdd'].rolling(window=14, min_periods=1).mean()
    
    # 30-day rolling averages
    region_df['hdd_30'] = region_df['hdd'].rolling(window=30, min_periods=1).mean()
    region_df['cdd_30'] = region_df['cdd'].rolling(window=30, min_periods=1).mean()
    
    degree_days_list.append(region_df)

# Combine all regions
degree_days_df = pd.concat(degree_days_list, ignore_index=True)

# Weekly aggregation
print("\n📅 Aggregating to Weekly Sums...")
degree_days_df['week'] = degree_days_df['date'].dt.to_period('W').dt.to_timestamp()

weekly_agg = degree_days_df.groupby(['region', 'week']).agg({
    'hdd': 'sum',
    'cdd': 'sum',
    'hdd_7': 'mean',
    'cdd_7': 'mean',
    'hdd_14': 'mean',
    'cdd_14': 'mean',
    'hdd_30': 'mean',
    'cdd_30': 'mean',
    'temp_avg_f': 'mean',
    'temp_avg_c': 'mean',
    'latitude': 'first',
    'longitude': 'first'
}).reset_index()

# Rename columns for clarity
weekly_agg.columns = [
    'region', 'week', 
    'hdd_weekly_sum', 'cdd_weekly_sum',
    'hdd_7day_avg', 'cdd_7day_avg',
    'hdd_14day_avg', 'cdd_14day_avg',
    'hdd_30day_avg', 'cdd_30day_avg',
    'temp_avg_f', 'temp_avg_c',
    'latitude', 'longitude'
]

print(f"✓ Generated {len(weekly_agg)} weekly records")
print(f"✓ Week range: {weekly_agg['week'].min().date()} to {weekly_agg['week'].max().date()}")

# Save daily degree days with rolling averages
print("\n💾 Saving Results...")

# Daily data
daily_output = degree_days_df[[
    'date', 'region', 'latitude', 'longitude',
    'temp_avg_c', 'temp_avg_f',
    'hdd', 'cdd',
    'hdd_7', 'cdd_7',
    'hdd_14', 'cdd_14',
    'hdd_30', 'cdd_30'
]].copy()

daily_output.to_csv('cleaned_data/degree_days_daily.csv', index=False)
print("✓ Saved: cleaned_data/degree_days_daily.csv")

# Weekly aggregated data
weekly_agg.to_csv('cleaned_data/degree_days.csv', index=False)
print("✓ Saved: cleaned_data/degree_days.csv")

# Generate summary statistics
print("\n" + "="*70)
print("SUMMARY STATISTICS")
print("="*70)

print("\n📊 Daily Degree Days by Region:")
summary = degree_days_df.groupby('region').agg({
    'hdd': ['mean', 'max'],
    'cdd': ['mean', 'max'],
    'temp_avg_f': 'mean'
}).round(2)
summary.columns = ['HDD_mean', 'HDD_max', 'CDD_mean', 'CDD_max', 'Temp_avg_F']
print(summary.to_string())

print("\n📊 Weekly Aggregated Degree Days (All Regions Combined):")
print(f"  Total weeks: {len(weekly_agg)}")
print(f"  HDD weekly sum - Mean: {weekly_agg['hdd_weekly_sum'].mean():.2f}, Max: {weekly_agg['hdd_weekly_sum'].max():.2f}")
print(f"  CDD weekly sum - Mean: {weekly_agg['cdd_weekly_sum'].mean():.2f}, Max: {weekly_agg['cdd_weekly_sum'].max():.2f}")

print("\n📊 Rolling Average Ranges:")
print(f"  7-day HDD avg: {weekly_agg['hdd_7day_avg'].min():.2f} to {weekly_agg['hdd_7day_avg'].max():.2f}")
print(f"  7-day CDD avg: {weekly_agg['cdd_7day_avg'].min():.2f} to {weekly_agg['cdd_7day_avg'].max():.2f}")
print(f"  14-day HDD avg: {weekly_agg['hdd_14day_avg'].min():.2f} to {weekly_agg['hdd_14day_avg'].max():.2f}")
print(f"  14-day CDD avg: {weekly_agg['cdd_14day_avg'].min():.2f} to {weekly_agg['cdd_14day_avg'].max():.2f}")
print(f"  30-day HDD avg: {weekly_agg['hdd_30day_avg'].min():.2f} to {weekly_agg['hdd_30day_avg'].max():.2f}")
print(f"  30-day CDD avg: {weekly_agg['cdd_30day_avg'].min():.2f} to {weekly_agg['cdd_30day_avg'].max():.2f}")

# Identify extreme weather weeks
print("\n🌡️  Top 5 Coldest Weeks (Highest HDD):")
coldest = weekly_agg.nlargest(5, 'hdd_weekly_sum')[['week', 'region', 'hdd_weekly_sum', 'temp_avg_f']]
print(coldest.to_string(index=False))

print("\n🌡️  Top 5 Hottest Weeks (Highest CDD):")
hottest = weekly_agg.nlargest(5, 'cdd_weekly_sum')[['week', 'region', 'cdd_weekly_sum', 'temp_avg_f']]
print(hottest.to_string(index=False))

print("\n" + "="*70)
print("✅ DEGREE DAYS CALCULATION COMPLETE!")
print("="*70)
print("\n📁 Output Files:")
print("  1. degree_days_daily.csv - Daily HDD/CDD with rolling averages")
print("  2. degree_days.csv - Weekly aggregated HDD/CDD")
print("\n🔍 Features Generated:")
print("  • hdd, cdd - Daily/Weekly degree days")
print("  • hdd_7, cdd_7 - 7-day rolling averages")
print("  • hdd_14, cdd_14 - 14-day rolling averages")
print("  • hdd_30, cdd_30 - 30-day rolling averages")
print("  • Weekly sums for model training")
print("\n✨ Ready for feature engineering and model training!")
print("="*70 + "\n")
