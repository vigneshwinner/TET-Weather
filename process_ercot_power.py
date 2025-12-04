"""
Process ERCOT hourly settlement prices into weekly Power_3yr.csv format
Matches the format of other commodity price files
"""

import pandas as pd
import numpy as np

print("="*80)
print("PROCESSING ERCOT POWER PRICES")
print("="*80)

# Load ERCOT hourly data
print("\n📂 Loading ERCOT hourly data...")
ercot_df = pd.read_csv('cleaned_data/ERCOT_hubs_Settlement_Price_2022 -2025.csv')

print(f"  ✓ Loaded {len(ercot_df):,} hourly records")
print(f"  Date range: {ercot_df['Delivery Date'].min()} to {ercot_df['Delivery Date'].max()}")

# Parse dates
ercot_df['Delivery Date'] = pd.to_datetime(ercot_df['Delivery Date'])

# Filter to main hub (HB_HUBAVG - the average of all major hubs)
print("\n  Using HB_HUBAVG (hub average) for Power prices...")
hub_df = ercot_df[ercot_df['Settlement Point Name'] == 'HB_HUBAVG'].copy()

print(f"  ✓ Filtered to {len(hub_df):,} HB_HUBAVG records")

# Calculate daily averages
print("\n📊 Calculating daily averages...")
daily_df = hub_df.groupby('Delivery Date').agg({
    'Settlement Point Price': ['mean', 'max', 'min', 'first']
}).reset_index()

daily_df.columns = ['Date', 'Close', 'High', 'Low', 'Open']

# Sort by date
daily_df = daily_df.sort_values('Date').reset_index(drop=True)

# Add volume column (set to empty string to match other files)
daily_df['Volume'] = ''

# Reorder columns to match other commodity files
daily_df = daily_df[['Date', 'Close', 'High', 'Low', 'Open', 'Volume']]

print(f"  ✓ Created {len(daily_df)} daily records")
print(f"  Date range: {daily_df['Date'].min().date()} to {daily_df['Date'].max().date()}")

# Add header rows to match other commodity files
header1 = "Power Futures Historical Data"
header2 = "From ERCOT Hub Average Settlement Prices"

# Save to Power_3yr.csv
output_file = 'cleaned_data/Power_3yr.csv'

with open(output_file, 'w') as f:
    f.write(f"{header1}\n")
    f.write(f"{header2}\n")
    daily_df.to_csv(f, index=False)

print(f"\n💾 Saved: {output_file}")

# Show summary statistics
print(f"\n📈 Price Statistics:")
print(f"  Mean:   ${daily_df['Close'].mean():.2f}")
print(f"  Median: ${daily_df['Close'].median():.2f}")
print(f"  Min:    ${daily_df['Close'].min():.2f}")
print(f"  Max:    ${daily_df['Close'].max():.2f}")
print(f"  Std:    ${daily_df['Close'].std():.2f}")

print("\n" + "="*80)
print("✅ ERCOT POWER PROCESSING COMPLETE!")
print("="*80 + "\n")
