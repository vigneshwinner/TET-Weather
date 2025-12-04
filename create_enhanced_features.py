"""
Enhanced Feature Engineering for Commodity Price Prediction
Adds price-based features: lagged returns, momentum, volatility, technical indicators
"""

import pandas as pd
import numpy as np
from pathlib import Path

print("="*80)
print("ENHANCED FEATURE ENGINEERING")
print("="*80)

# ============================================================================
# Load Price Data
# ============================================================================

print(f"\n📂 Loading price data...")

price_files = {
    'Brent': 'cleaned_data/Brent_3yr.csv',
    'Henry_Hub': 'cleaned_data/Henry_Hub_3yr.csv',
    'Power': 'cleaned_data/Power_3yr.csv',
    'Copper': 'cleaned_data/Copper_3yr.csv',
    'Corn': 'cleaned_data/Corn_3yr.csv'
}

all_features = []

for commodity, filepath in price_files.items():
    print(f"\n  Processing {commodity}...")
    
    df = pd.read_csv(filepath, skiprows=2)
    df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Convert volume strings (e.g., "146.02K" -> 146020)
    if df['Volume'].dtype == 'object':
        df['Volume'] = df['Volume'].astype(str).str.replace('K', 'e3').str.replace('M', 'e6').str.replace('B', 'e9')
        df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')
    
    df = df.sort_values('Date').reset_index(drop=True)
    
    # ========================================================================
    # 1. Price-based Features
    # ========================================================================
    
    # Log returns at multiple horizons
    df['return_1d'] = np.log(df['Close'] / df['Close'].shift(1))
    df['return_2d'] = np.log(df['Close'] / df['Close'].shift(2))
    df['return_3d'] = np.log(df['Close'] / df['Close'].shift(3))
    df['return_5d'] = np.log(df['Close'] / df['Close'].shift(5))
    df['return_10d'] = np.log(df['Close'] / df['Close'].shift(10))
    df['return_20d'] = np.log(df['Close'] / df['Close'].shift(20))
    
    # Lagged returns (shifted 1 day to avoid lookahead)
    df['return_lag1'] = df['return_1d'].shift(1)
    df['return_lag2'] = df['return_1d'].shift(2)
    df['return_lag3'] = df['return_1d'].shift(3)
    df['return_lag5'] = df['return_1d'].shift(5)
    df['return_lag10'] = df['return_1d'].shift(10)
    
    # ========================================================================
    # 2. Volatility Features
    # ========================================================================
    
    # Rolling volatility (standard deviation of returns)
    df['volatility_5d'] = df['return_1d'].rolling(5, min_periods=2).std()
    df['volatility_10d'] = df['return_1d'].rolling(10, min_periods=5).std()
    df['volatility_20d'] = df['return_1d'].rolling(20, min_periods=10).std()
    df['volatility_30d'] = df['return_1d'].rolling(30, min_periods=15).std()
    
    # Parkinson volatility (uses high-low range, more efficient estimator)
    df['parkinson_vol_10d'] = np.sqrt(
        (np.log(df['High'] / df['Low']) ** 2).rolling(10, min_periods=5).mean() / (4 * np.log(2))
    )
    
    # Volatility of volatility
    df['vol_of_vol_20d'] = df['volatility_5d'].rolling(20, min_periods=10).std()
    
    # ========================================================================
    # 3. Momentum & Trend Features
    # ========================================================================
    
    # Moving averages
    df['ma_5d'] = df['Close'].rolling(5, min_periods=2).mean()
    df['ma_10d'] = df['Close'].rolling(10, min_periods=5).mean()
    df['ma_20d'] = df['Close'].rolling(20, min_periods=10).mean()
    df['ma_50d'] = df['Close'].rolling(50, min_periods=25).mean()
    
    # Price relative to moving averages
    df['price_to_ma5'] = df['Close'] / df['ma_5d'] - 1
    df['price_to_ma10'] = df['Close'] / df['ma_10d'] - 1
    df['price_to_ma20'] = df['Close'] / df['ma_20d'] - 1
    df['price_to_ma50'] = df['Close'] / df['ma_50d'] - 1
    
    # Moving average crossovers
    df['ma5_to_ma20'] = df['ma_5d'] / df['ma_20d'] - 1
    df['ma10_to_ma50'] = df['ma_10d'] / df['ma_50d'] - 1
    
    # RSI (Relative Strength Index)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14, min_periods=7).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14, min_periods=7).mean()
    rs = gain / loss
    df['rsi_14d'] = 100 - (100 / (1 + rs))
    
    # ========================================================================
    # 4. Price Range & High-Low Features
    # ========================================================================
    
    # Daily range
    df['daily_range'] = (df['High'] - df['Low']) / df['Close']
    df['range_5d_avg'] = df['daily_range'].rolling(5, min_periods=2).mean()
    df['range_20d_avg'] = df['daily_range'].rolling(20, min_periods=10).mean()
    
    # Position within daily range
    df['close_position'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'] + 1e-8)
    
    # Distance from recent high/low
    df['dist_from_high_20d'] = df['Close'] / df['Close'].rolling(20, min_periods=10).max() - 1
    df['dist_from_low_20d'] = df['Close'] / df['Close'].rolling(20, min_periods=10).min() - 1
    
    # ========================================================================
    # 5. Volume Features (if available)
    # ========================================================================
    
    if df['Volume'].notna().sum() > 0:
        df['volume_5d_avg'] = df['Volume'].rolling(5, min_periods=2).mean()
        df['volume_20d_avg'] = df['Volume'].rolling(20, min_periods=10).mean()
        df['volume_ratio'] = df['Volume'] / (df['volume_20d_avg'] + 1)
        
        # Price-volume correlation
        df['price_volume_corr_20d'] = df['return_1d'].rolling(20, min_periods=10).corr(df['Volume'])
    else:
        df['volume_5d_avg'] = 0
        df['volume_20d_avg'] = 0
        df['volume_ratio'] = 0
        df['price_volume_corr_20d'] = 0
    
    # ========================================================================
    # 6. Regime Features
    # ========================================================================
    
    # Bull/bear regime (20d return positive/negative)
    df['regime_bull'] = (df['return_20d'] > 0).astype(int)
    
    # High/low volatility regime
    vol_median = df['volatility_20d'].median()
    df['regime_high_vol'] = (df['volatility_20d'] > vol_median).astype(int)
    
    # Trending vs mean-reverting
    # Hurst exponent approximation: if returns are trending, consecutive returns have same sign
    df['trend_strength'] = df['return_1d'].rolling(10, min_periods=5).apply(
        lambda x: np.abs(x.autocorr()) if len(x) > 2 else 0
    )
    
    # ========================================================================
    # 7. Seasonality Features
    # ========================================================================
    
    df['month'] = df['Date'].dt.month
    df['quarter'] = df['Date'].dt.quarter
    df['day_of_week'] = df['Date'].dt.dayofweek
    df['week_of_year'] = df['Date'].dt.isocalendar().week
    
    # Cyclical encoding for month and day of week
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    # ========================================================================
    # 8. Target Variable
    # ========================================================================
    
    # Next-day return (what we're trying to predict)
    df['return_next'] = df['return_1d'].shift(-1)
    df['direction_next'] = (df['return_next'] > 0).astype(int)
    
    # ========================================================================
    # Weekly Aggregation
    # ========================================================================
    
    df['week'] = df['Date'] - pd.to_timedelta(df['Date'].dt.dayofweek, unit='D')
    df['commodity'] = commodity
    
    # Keep only feature columns for aggregation
    feature_cols = [col for col in df.columns if col not in 
                    ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
    
    # Aggregate to weekly (mean for most features, last for target)
    agg_dict = {col: 'mean' for col in feature_cols if col not in ['week', 'commodity', 'return_next', 'direction_next']}
    agg_dict['return_next'] = 'last'
    agg_dict['direction_next'] = 'last'
    
    weekly_df = df.groupby(['week', 'commodity']).agg(agg_dict).reset_index()
    
    all_features.append(weekly_df)
    print(f"    Created {len(feature_cols)} features -> {len(weekly_df)} weekly records")

# ============================================================================
# Combine All Commodities
# ============================================================================

print(f"\n🔗 Combining all commodities...")
enhanced_features_df = pd.concat(all_features, ignore_index=True)

# Remove rows with missing targets
enhanced_features_df = enhanced_features_df[enhanced_features_df['return_next'].notna()].copy()

print(f"  ✓ Total records: {len(enhanced_features_df)}")
print(f"  ✓ Commodities: {enhanced_features_df['commodity'].unique().tolist()}")
print(f"  ✓ Date range: {enhanced_features_df['week'].min().date()} to {enhanced_features_df['week'].max().date()}")

# ============================================================================
# Save Enhanced Features
# ============================================================================

output_file = 'cleaned_data/enhanced_price_features.csv'
enhanced_features_df.to_csv(output_file, index=False)
print(f"\n💾 Saved: {output_file}")

# Feature summary
feature_cols_out = [col for col in enhanced_features_df.columns if col not in ['week', 'commodity', 'return_next', 'direction_next']]
print(f"\n📊 Feature Summary:")
print(f"  Total features: {len(feature_cols_out)}")
print(f"  Categories:")
print(f"    • Lagged returns: {len([c for c in feature_cols_out if 'return' in c])}")
print(f"    • Volatility: {len([c for c in feature_cols_out if 'vol' in c or 'parkinson' in c])}")
print(f"    • Momentum/MA: {len([c for c in feature_cols_out if 'ma' in c or 'rsi' in c])}")
print(f"    • Range/High-Low: {len([c for c in feature_cols_out if 'range' in c or 'dist_from' in c or 'close_position' in c])}")
print(f"    • Volume: {len([c for c in feature_cols_out if 'volume' in c])}")
print(f"    • Regime: {len([c for c in feature_cols_out if 'regime' in c or 'trend_strength' in c])}")
print(f"    • Seasonality: {len([c for c in feature_cols_out if any(x in c for x in ['month', 'quarter', 'dow', 'week_of_year'])])}")

print("\n" + "="*80)
print("✅ ENHANCED FEATURES CREATED!")
print("="*80)
