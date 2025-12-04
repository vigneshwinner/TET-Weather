"""
Fix Corn_3yr.csv to match the format of other commodity files (with 2 header rows)
"""

import pandas as pd

print("Fixing Corn_3yr.csv format...")

# Read the current file (skip no rows since it doesn't have headers)
df = pd.read_csv('cleaned_data/Corn_3yr_backup.csv')

print(f"Loaded {len(df)} records")

# Add header rows
header1 = "Price,Close,High,Low,Open,Volume"
header2 = "Ticker,ZC=F,ZC=F,ZC=F,ZC=F,ZC=F"

# Save with proper headers
with open('cleaned_data/Corn_3yr.csv', 'w') as f:
    f.write(f"{header1}\n")
    f.write(f"{header2}\n")
    df.to_csv(f, index=False)

print("✓ Corn_3yr.csv updated with proper headers")
