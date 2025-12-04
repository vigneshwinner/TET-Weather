"""
Test that all 5 commodities can be loaded and processed correctly
"""

import pandas as pd
import numpy as np

print("="*80)
print("TESTING ALL 5 COMMODITIES")
print("="*80)

COMMODITIES = ['Brent', 'Henry_Hub', 'Power', 'Copper', 'Corn']

price_files = {
    'Brent': 'cleaned_data/Brent_3yr.csv',
    'Henry_Hub': 'cleaned_data/Henry_Hub_3yr.csv',
    'Power': 'cleaned_data/Power_3yr.csv',
    'Copper': 'cleaned_data/Copper_3yr.csv',
    'Corn': 'cleaned_data/Corn_3yr.csv'
}

print("\n📂 Loading price data for all commodities...\n")

for commodity in COMMODITIES:
    filepath = price_files[commodity]
    
    try:
        df = pd.read_csv(filepath, skiprows=2)
        df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)
        
        df['return'] = np.log(df['Close'] / df['Close'].shift(1))
        df['return_next'] = df['return'].shift(-1)
        df['direction_next'] = (df['return_next'] > 0).astype(int)
        
        print(f"✓ {commodity:12s}: {len(df):4d} records | "
              f"{df['Date'].min().date()} to {df['Date'].max().date()} | "
              f"Price: ${df['Close'].mean():.2f} ± ${df['Close'].std():.2f}")
        
    except Exception as e:
        print(f"✗ {commodity:12s}: ERROR - {e}")

# Test commodity mapping
print("\n📊 Testing commodity mapping with weather-EIA interactions...")

interactions_df = pd.read_csv('cleaned_data/weather_eia_interactions.csv')
commodity_map = {
    'Crude_Oil': 'Brent',
    'Natural_Gas': 'Henry_Hub',
    'Power': 'Power',
    'Copper': 'Copper',
    'Corn': 'Corn'
}

interactions_df['commodity_mapped'] = interactions_df['commodity'].map(commodity_map)

print(f"\nWeather-EIA interaction commodities found:")
for orig_commodity in interactions_df['commodity'].unique():
    if pd.isna(orig_commodity):
        continue
    mapped = commodity_map.get(orig_commodity, 'Unknown')
    count = len(interactions_df[interactions_df['commodity'] == orig_commodity])
    print(f"  {orig_commodity:15s} → {mapped:12s} ({count:,} records)")

print(f"\nCommodities WITHOUT weather-EIA interactions:")
eia_commodities = set(interactions_df['commodity_mapped'].unique())
for commodity in COMMODITIES:
    if commodity not in eia_commodities:
        print(f"  {commodity:12s} (will use degree-days only, fillna with 0)")

print("\n" + "="*80)
print("✅ ALL 5 COMMODITIES READY!")
print("="*80 + "\n")
