"""
Stress Test Analysis
Evaluate model robustness during major market events
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("STRESS TEST ANALYSIS - MAJOR MARKET EVENTS")
print("="*80)

# ============================================================================
# Event Definitions - Using available data range
# ============================================================================

# Since signals only cover 2024-2025, we'll analyze different regimes
# instead of historical events

print(f"\n📂 Loading backtest results and signals...")

# Load signals
signals_file = Path('reports/signals/weekly_signals.csv')
if not signals_file.exists():
    print(f"  ⚠️  Signals file not found: {signals_file}")
    print(f"  Please run signal_generator.py first")
    exit(1)

signals_df = pd.read_csv(signals_file)
signals_df['date'] = pd.to_datetime(signals_df['week'])
signals_df = signals_df.rename(columns={'raw_signal': 'signal', 'y_true': 'actual_return'})
signals_df['model'] = 'ridge_baseline'

# Merge - signals already have returns
backtest_df = signals_df[['date', 'commodity', 'model', 'signal', 'actual_return']].copy()

print(f"  ✓ Loaded {len(backtest_df):,} records")
print(f"  Models: {backtest_df['model'].unique().tolist()}")
print(f"  Date range: {backtest_df['date'].min().date()} to {backtest_df['date'].max().date()}")

# Load SSI data if available
ssi_file = Path('cleaned_data/ssi_results.csv')
if ssi_file.exists():
    ssi_df = pd.read_csv(ssi_file)
    ssi_df['date'] = pd.to_datetime(ssi_df['date'])
    print(f"  ✓ Loaded SSI data: {len(ssi_df):,} records")
    has_ssi = True
else:
    print(f"  ⚠️  SSI data not found at {ssi_file}")
    has_ssi = False

# Load price data for overlays
price_files = {
    'Brent': 'cleaned_data/Brent_3yr.csv',
    'Henry_Hub': 'cleaned_data/Henry_Hub_3yr.csv',
    'Power': 'cleaned_data/Power_3yr.csv',
    'Copper': 'cleaned_data/Copper_3yr.csv',
    'Corn': 'cleaned_data/Corn_3yr.csv'
}

price_data = []
for commodity, filepath in price_files.items():
    if Path(filepath).exists():
        df = pd.read_csv(filepath, skiprows=2)
        df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        df['Date'] = pd.to_datetime(df['Date'])
        df['commodity'] = commodity
        price_data.append(df[['Date', 'commodity', 'Close']])

prices_df = pd.concat(price_data, ignore_index=True)
prices_df = prices_df.rename(columns={'Date': 'date'})

print(f"  ✓ Loaded price data for {len(price_data)} commodities")

# ============================================================================
# Classify Regimes
# ============================================================================

print(f"\n📊 Classifying market regimes based on volatility...")

# Define regimes based on volatility
backtest_df = backtest_df.sort_values(['commodity', 'date'])
backtest_df['volatility_20w'] = backtest_df.groupby('commodity')['actual_return'].transform(
    lambda x: x.rolling(20, min_periods=5).std()
)

# Classify into regimes
vol_median = backtest_df['volatility_20w'].median()
vol_75 = backtest_df['volatility_20w'].quantile(0.75)

backtest_df['regime'] = pd.cut(
    backtest_df['volatility_20w'], 
    bins=[0, vol_median, vol_75, np.inf],
    labels=['low_vol', 'medium_vol', 'high_vol']
)

print(f"\n  Volatility thresholds:")
print(f"    Low/Medium:   {vol_median:.4f}")
print(f"    Medium/High:  {vol_75:.4f}")

print(f"\n  Regime distribution:")
regime_counts = backtest_df['regime'].value_counts().sort_index()
for regime, count in regime_counts.items():
    pct = count / len(backtest_df) * 100
    print(f"    {regime:15s}: {count:4d} records ({pct:.1f}%)")

# ============================================================================
# Regime Analysis
# ============================================================================

print(f"\n" + "="*80)
print("REGIME ANALYSIS - PERFORMANCE ACROSS MARKET CONDITIONS")
print("="*80)

regime_results = []

for regime in ['low_vol', 'medium_vol', 'high_vol']:
    regime_data = backtest_df[backtest_df['regime'] == regime].copy()
    
    if len(regime_data) == 0:
        continue
    
    print(f"\n{'='*80}")
    print(f"REGIME: {regime.upper().replace('_', ' ')}")
    print(f"Period: {regime_data['date'].min().date()} to {regime_data['date'].max().date()}")
    print(f"{'='*80}")
    
    print(f"\n  Data: {len(regime_data)} records")
    
    # Calculate regime metrics by model
    for model in regime_data['model'].unique():
        model_data = regime_data[regime_data['model'] == model].copy()
        
        # Calculate returns (signal * actual_return)
        model_data['position_return'] = model_data['signal'] * model_data['actual_return']
        
        # Portfolio-level (equal weight across commodities)
        portfolio_returns = model_data.groupby('date')['position_return'].mean()
        
        # Metrics
        total_return = portfolio_returns.sum()
        cum_returns = (1 + portfolio_returns).cumprod()
        running_max = np.maximum.accumulate(cum_returns)
        drawdown = (cum_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        hit_ratio = (portfolio_returns > 0).sum() / len(portfolio_returns) if len(portfolio_returns) > 0 else 0
        
        # Volatility and Sharpe
        regime_vol = portfolio_returns.std() * np.sqrt(52)
        regime_sharpe = (portfolio_returns.mean() * 52) / regime_vol if regime_vol > 0 else 0
        
        # Exposure
        exposure = (model_data['signal'].abs() > 0).sum() / len(model_data)
        
        print(f"\n  📊 {model.upper()}")
        print(f"    Regime Return:   {total_return*100:>8.2f}%")
        print(f"    Sharpe Ratio:    {regime_sharpe:>8.2f}")
        print(f"    Max Drawdown:    {max_drawdown*100:>8.2f}%")
        print(f"    Hit Ratio:       {hit_ratio*100:>8.2f}%")
        print(f"    Exposure:        {exposure*100:>8.2f}%")
        
        # Store results
        regime_results.append({
            'regime': regime,
            'model': model,
            'regime_return': total_return,
            'regime_sharpe': regime_sharpe,
            'max_drawdown': max_drawdown,
            'hit_ratio': hit_ratio,
            'exposure': exposure,
            'n_weeks': len(portfolio_returns)
        })
        
        # By commodity
        print(f"\n    By Commodity:")
        for commodity in model_data['commodity'].unique():
            comm_data = model_data[model_data['commodity'] == commodity]
            comm_return = (comm_data['signal'] * comm_data['actual_return']).sum()
            comm_hit = ((comm_data['signal'] * comm_data['actual_return']) > 0).sum() / len(comm_data)
            
            regime_results.append({
                'regime': regime,
                'model': model,
                'commodity': commodity,
                'regime_return': comm_return,
                'hit_ratio': comm_hit,
                'n_records': len(comm_data)
            })
            
            print(f"      {commodity:12s}: {comm_return*100:>6.2f}% return, {comm_hit*100:>5.1f}% hit rate")

# Save results
regime_results_df = pd.DataFrame(regime_results)
output_file = 'stress_test_report.csv'
regime_results_df.to_csv(output_file, index=False)
print(f"\n💾 Saved: {output_file}")

# ============================================================================
# Visualization
# ============================================================================

print(f"\n📊 Creating stress test visualizations...")

output_dir = Path('cleaned_data/stress_test_plots')
output_dir.mkdir(exist_ok=True)

sns.set_style("whitegrid")

# 1. Regime Performance Comparison
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Performance Across Market Regimes', fontsize=16, fontweight='bold')

# Filter to portfolio-level results (no commodity column)
portfolio_results = regime_results_df[regime_results_df['commodity'].isna()].copy()

if len(portfolio_results) > 0:
    # Regime returns by model
    ax = axes[0, 0]
    pivot_returns = portfolio_results.pivot(index='regime', columns='model', values='regime_return') * 100
    pivot_returns.plot(kind='bar', ax=ax, width=0.8)
    ax.set_title('Regime Returns by Model')
    ax.set_xlabel('Regime')
    ax.set_ylabel('Return (%)')
    ax.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    
    # Sharpe ratios
    ax = axes[0, 1]
    pivot_sharpe = portfolio_results.pivot(index='regime', columns='model', values='regime_sharpe')
    pivot_sharpe.plot(kind='bar', ax=ax, width=0.8)
    ax.set_title('Regime Sharpe Ratios')
    ax.set_xlabel('Regime')
    ax.set_ylabel('Sharpe Ratio')
    ax.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    
    # Max drawdown
    ax = axes[1, 0]
    pivot_dd = portfolio_results.pivot(index='regime', columns='model', values='max_drawdown') * 100
    pivot_dd.plot(kind='bar', ax=ax, width=0.8)
    ax.set_title('Maximum Drawdown During Regimes')
    ax.set_xlabel('Regime')
    ax.set_ylabel('Max Drawdown (%)')
    ax.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Hit ratio
    ax = axes[1, 1]
    pivot_hit = portfolio_results.pivot(index='regime', columns='model', values='hit_ratio') * 100
    pivot_hit.plot(kind='bar', ax=ax, width=0.8)
    ax.set_title('Hit Ratio During Regimes')
    ax.set_xlabel('Regime')
    ax.set_ylabel('Hit Ratio (%)')
    ax.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=50, color='gray', linestyle='--', linewidth=1, alpha=0.5)

plt.tight_layout()
plt.savefig(output_dir / 'regime_performance_comparison.png', dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: {output_dir / 'regime_performance_comparison.png'}")
plt.close()

# 2. Commodity-specific performance across regimes
commodity_results = regime_results_df[regime_results_df['commodity'].notna()].copy()

if len(commodity_results) > 0:
    for regime in commodity_results['regime'].unique():
        regime_comm_data = commodity_results[commodity_results['regime'] == regime]
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(f'{regime.upper().replace("_", " ")} Regime - Commodity Performance', 
                     fontsize=14, fontweight='bold')
        
        # Returns
        ax = axes[0]
        pivot = regime_comm_data.pivot(index='commodity', columns='model', values='regime_return') * 100
        pivot.plot(kind='bar', ax=ax, width=0.8)
        ax.set_title('Returns by Commodity')
        ax.set_xlabel('Commodity')
        ax.set_ylabel('Return (%)')
        ax.legend(title='Model')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        
        # Hit ratio
        ax = axes[1]
        pivot_hit = regime_comm_data.pivot(index='commodity', columns='model', values='hit_ratio') * 100
        pivot_hit.plot(kind='bar', ax=ax, width=0.8)
        ax.set_title('Hit Ratio by Commodity')
        ax.set_xlabel('Commodity')
        ax.set_ylabel('Hit Ratio (%)')
        ax.legend(title='Model')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=50, color='gray', linestyle='--', linewidth=1)
        
        plt.tight_layout()
        filename = f"regime_{regime}_commodities.png"
        plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {output_dir / filename}")
        plt.close()

print(f"\n" + "="*80)
print("✅ STRESS TEST ANALYSIS COMPLETE!")
print(f"📁 Report: {output_file}")
print(f"📊 Plots: {output_dir}/")
print("="*80 + "\n")
