"""
Vectorized Weekly Backtester
Consumes trading signals and market returns to produce PnL and summary statistics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("BACKTESTING FRAMEWORK - VECTORIZED WEEKLY BACKTEST")
print("="*80)

# ============================================================================
# Configuration
# ============================================================================

TRANSACTION_COST = 0.001  # 0.1% per trade (bid-ask spread + commissions)
SLIPPAGE = 0.0005  # 0.05% slippage
INITIAL_CAPITAL = 100000

# Risk weighting options
WEIGHTING = 'equal_weight'  # 'equal_weight' or 'equal_risk'

print(f"\n⚙️  Configuration:")
print(f"  Initial Capital: ${INITIAL_CAPITAL:,.0f}")
print(f"  Transaction Cost: {TRANSACTION_COST*100:.2f}%")
print(f"  Slippage: {SLIPPAGE*100:.3f}%")
print(f"  Position Weighting: {WEIGHTING}")

# ============================================================================
# Load Data
# ============================================================================

print(f"\n📂 Loading signals and returns...")

# Load signals from signal generator
signals_file = Path('reports/signals/weekly_signals.csv')
if not signals_file.exists():
    print(f"  ⚠️  Signals file not found: {signals_file}")
    print(f"  Please run signal_generator.py first")
    exit(1)

# Load signals
signals_df = pd.read_csv(signals_file)
signals_df['date'] = pd.to_datetime(signals_df['week'])
signals_df = signals_df.rename(columns={'raw_signal': 'signal', 'y_true': 'actual_return'})

# Add model column (assuming baseline Ridge model)
signals_df['model'] = 'ridge_baseline'

print(f"  ✓ Loaded signals: {len(signals_df):,} records")
print(f"  Models: {signals_df['model'].unique().tolist()}")
print(f"  Commodities: {signals_df['commodity'].unique().tolist()}")

# Load actual returns (from predictions)
baseline_dir = Path('cleaned_data/model_artifacts')
predictions_df = pd.read_csv(baseline_dir / 'predictions.csv')
predictions_df['date'] = pd.to_datetime(predictions_df['date'])

# Also load XGBoost predictions
xgb_dir = Path('cleaned_data/xgboost_artifacts')
xgb_predictions_df = pd.read_csv(xgb_dir / 'xgb_predictions.csv')
xgb_predictions_df['date'] = pd.to_datetime(xgb_predictions_df['date'])

print(f"  ✓ Loaded returns: {len(predictions_df):,} baseline, {len(xgb_predictions_df):,} XGBoost")

# ============================================================================
# Merge Signals with Returns
# ============================================================================

print(f"\n🔗 Merging signals with actual returns...")

# Signals already have actual returns, so we don't need to merge
backtest_df = signals_df.copy()

# Keep only needed columns
backtest_df = backtest_df[['date', 'commodity', 'model', 'signal', 'actual_return']].copy()

print(f"  ✓ Dataset ready: {len(backtest_df):,} records")

# ============================================================================
# Calculate PnL
# ============================================================================

print(f"\n💰 Calculating PnL with transaction costs...")

# Calculate position changes (for turnover and transaction costs)
backtest_df = backtest_df.sort_values(['model', 'commodity', 'date'])
backtest_df['signal_lag'] = backtest_df.groupby(['model', 'commodity'])['signal'].shift(1)
backtest_df['position_change'] = (backtest_df['signal'] != backtest_df['signal_lag']).astype(int)

# First position is always a change
backtest_df.loc[backtest_df['signal_lag'].isna(), 'position_change'] = 1

# Calculate gross return (signal * actual_return)
backtest_df['gross_return'] = backtest_df['signal'] * backtest_df['actual_return']

# Apply transaction costs only when position changes
backtest_df['transaction_cost'] = backtest_df['position_change'] * (TRANSACTION_COST + SLIPPAGE)

# Net return after costs
backtest_df['net_return'] = backtest_df['gross_return'] - backtest_df['transaction_cost']

# Calculate weights
if WEIGHTING == 'equal_weight':
    # Equal weight across commodities
    backtest_df['weight'] = 1.0 / backtest_df.groupby(['model', 'date'])['commodity'].transform('count')
elif WEIGHTING == 'equal_risk':
    # Weight inversely proportional to volatility
    volatility = backtest_df.groupby(['model', 'commodity'])['actual_return'].transform(lambda x: x.rolling(52, min_periods=1).std())
    inv_vol = 1.0 / volatility.replace(0, np.nan)
    backtest_df['weight'] = inv_vol / backtest_df.groupby(['model', 'date'])['commodity'].transform(lambda x: inv_vol.sum())
else:
    backtest_df['weight'] = 1.0 / backtest_df.groupby(['model', 'date'])['commodity'].transform('count')

# Weighted returns
backtest_df['weighted_return'] = backtest_df['net_return'] * backtest_df['weight']

print(f"  ✓ PnL calculated with {WEIGHTING} weighting")

# ============================================================================
# Portfolio-Level Metrics
# ============================================================================

print(f"\n📊 Computing portfolio metrics...")

# Aggregate to portfolio level (sum across commodities for each model/date)
portfolio_returns = backtest_df.groupby(['model', 'date']).agg({
    'weighted_return': 'sum',
    'position_change': 'sum',
    'transaction_cost': 'sum'
}).reset_index()

portfolio_returns = portfolio_returns.rename(columns={'weighted_return': 'portfolio_return'})

# Calculate cumulative returns
portfolio_returns = portfolio_returns.sort_values(['model', 'date'])
portfolio_returns['cum_return'] = portfolio_returns.groupby('model')['portfolio_return'].cumsum()
portfolio_returns['equity'] = INITIAL_CAPITAL * (1 + portfolio_returns['cum_return'])

# Calculate metrics per model
results = []

for model in portfolio_returns['model'].unique():
    model_data = portfolio_returns[portfolio_returns['model'] == model].copy()
    
    returns = model_data['portfolio_return'].values
    equity = model_data['equity'].values
    
    # Basic metrics
    total_return = (equity[-1] - INITIAL_CAPITAL) / INITIAL_CAPITAL
    n_weeks = len(returns)
    n_years = n_weeks / 52
    cagr = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0
    
    # Volatility
    annualized_vol = returns.std() * np.sqrt(52)
    
    # Sharpe ratio (assume 0% risk-free rate)
    sharpe = (returns.mean() * 52) / annualized_vol if annualized_vol > 0 else 0
    
    # Sortino ratio (downside deviation)
    downside_returns = returns[returns < 0]
    downside_vol = downside_returns.std() * np.sqrt(52) if len(downside_returns) > 0 else annualized_vol
    sortino = (returns.mean() * 52) / downside_vol if downside_vol > 0 else 0
    
    # Max drawdown
    cum_returns = (1 + returns).cumprod()
    running_max = np.maximum.accumulate(cum_returns)
    drawdown = (cum_returns - running_max) / running_max
    max_drawdown = drawdown.min()
    
    # Average drawdown length
    in_drawdown = drawdown < 0
    drawdown_lengths = []
    current_length = 0
    for is_dd in in_drawdown:
        if is_dd:
            current_length += 1
        elif current_length > 0:
            drawdown_lengths.append(current_length)
            current_length = 0
    if current_length > 0:
        drawdown_lengths.append(current_length)
    avg_dd_length = np.mean(drawdown_lengths) if drawdown_lengths else 0
    
    # Hit ratio
    hit_ratio = (returns > 0).sum() / len(returns) if len(returns) > 0 else 0
    
    # Turnover (average position changes per week)
    turnover = model_data['position_change'].sum() / n_weeks if n_weeks > 0 else 0
    annual_turnover = turnover * 52
    
    # Rolling 52-week Sharpe
    model_data['rolling_return_mean'] = model_data['portfolio_return'].rolling(52, min_periods=12).mean()
    model_data['rolling_return_std'] = model_data['portfolio_return'].rolling(52, min_periods=12).std()
    model_data['rolling_sharpe'] = (model_data['rolling_return_mean'] * 52) / (model_data['rolling_return_std'] * np.sqrt(52))
    
    results.append({
        'model': model,
        'total_return': total_return,
        'cagr': cagr,
        'annualized_volatility': annualized_vol,
        'sharpe_ratio': sharpe,
        'sortino_ratio': sortino,
        'max_drawdown': max_drawdown,
        'avg_drawdown_length_weeks': avg_dd_length,
        'hit_ratio': hit_ratio,
        'weekly_turnover': turnover,
        'annual_turnover': annual_turnover,
        'n_weeks': n_weeks,
        'n_years': n_years
    })

results_df = pd.DataFrame(results)

# Save results
output_file = 'backtest_scorecard.csv'
results_df.to_csv(output_file, index=False)
print(f"\n💾 Saved: {output_file}")

# ============================================================================
# Display Results
# ============================================================================

print(f"\n" + "="*80)
print("BACKTEST RESULTS - PORTFOLIO PERFORMANCE")
print("="*80)

for _, row in results_df.iterrows():
    print(f"\n📈 {row['model'].upper()}")
    print(f"  Total Return:          {row['total_return']*100:>8.2f}%")
    print(f"  CAGR:                  {row['cagr']*100:>8.2f}%")
    print(f"  Annualized Volatility: {row['annualized_volatility']*100:>8.2f}%")
    print(f"  Sharpe Ratio:          {row['sharpe_ratio']:>8.2f}")
    print(f"  Sortino Ratio:         {row['sortino_ratio']:>8.2f}")
    print(f"  Max Drawdown:          {row['max_drawdown']*100:>8.2f}%")
    print(f"  Avg DD Length:         {row['avg_drawdown_length_weeks']:>8.1f} weeks")
    print(f"  Hit Ratio:             {row['hit_ratio']*100:>8.2f}%")
    print(f"  Annual Turnover:       {row['annual_turnover']:>8.1f}x")

# ============================================================================
# Visualization
# ============================================================================

print(f"\n📊 Creating visualizations...")

output_dir = Path('cleaned_data/backtest_plots')
output_dir.mkdir(exist_ok=True)

sns.set_style("whitegrid")

# 1. Equity Curves
fig, ax = plt.subplots(figsize=(14, 7))
for model in portfolio_returns['model'].unique():
    model_data = portfolio_returns[portfolio_returns['model'] == model]
    ax.plot(model_data['date'], model_data['equity'], label=model, linewidth=2, alpha=0.8)

ax.axhline(y=INITIAL_CAPITAL, color='gray', linestyle='--', linewidth=1, alpha=0.5)
ax.set_title('Portfolio Equity Curves by Model', fontsize=16, fontweight='bold')
ax.set_xlabel('Date')
ax.set_ylabel('Equity ($)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(output_dir / 'equity_curves.png', dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: {output_dir / 'equity_curves.png'}")
plt.close()

# 2. Cumulative Returns
fig, ax = plt.subplots(figsize=(14, 7))
for model in portfolio_returns['model'].unique():
    model_data = portfolio_returns[portfolio_returns['model'] == model]
    ax.plot(model_data['date'], model_data['cum_return'] * 100, label=model, linewidth=2, alpha=0.8)

ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
ax.set_title('Cumulative Returns by Model', fontsize=16, fontweight='bold')
ax.set_xlabel('Date')
ax.set_ylabel('Cumulative Return (%)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(output_dir / 'cumulative_returns.png', dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: {output_dir / 'cumulative_returns.png'}")
plt.close()

# 3. Rolling 52-Week Sharpe Ratio
fig, ax = plt.subplots(figsize=(14, 7))
for model in portfolio_returns['model'].unique():
    model_data = portfolio_returns[portfolio_returns['model'] == model]
    # Calculate rolling Sharpe
    model_data['rolling_sharpe'] = (
        model_data['portfolio_return'].rolling(52, min_periods=12).mean() * 52 / 
        (model_data['portfolio_return'].rolling(52, min_periods=12).std() * np.sqrt(52))
    )
    ax.plot(model_data['date'], model_data['rolling_sharpe'], label=model, linewidth=2, alpha=0.8)

ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
ax.axhline(y=1, color='green', linestyle='--', linewidth=1, alpha=0.5, label='Sharpe=1')
ax.set_title('Rolling 52-Week Sharpe Ratio', fontsize=16, fontweight='bold')
ax.set_xlabel('Date')
ax.set_ylabel('Sharpe Ratio')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(output_dir / 'rolling_sharpe.png', dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: {output_dir / 'rolling_sharpe.png'}")
plt.close()

# 4. Drawdown Chart
fig, ax = plt.subplots(figsize=(14, 7))
for model in portfolio_returns['model'].unique():
    model_data = portfolio_returns[portfolio_returns['model'] == model].copy()
    returns = model_data['portfolio_return'].values
    cum_returns = (1 + returns).cumprod()
    running_max = np.maximum.accumulate(cum_returns)
    drawdown = (cum_returns - running_max) / running_max
    model_data['drawdown'] = drawdown
    ax.fill_between(model_data['date'], drawdown * 100, 0, label=model, alpha=0.5)

ax.set_title('Drawdown Over Time', fontsize=16, fontweight='bold')
ax.set_xlabel('Date')
ax.set_ylabel('Drawdown (%)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(output_dir / 'drawdown.png', dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: {output_dir / 'drawdown.png'}")
plt.close()

# 5. Monthly Returns Heatmap (for first model)
first_model = portfolio_returns['model'].unique()[0]
model_data = portfolio_returns[portfolio_returns['model'] == first_model].copy()
model_data['year'] = model_data['date'].dt.year
model_data['month'] = model_data['date'].dt.month

monthly_returns = model_data.groupby(['year', 'month'])['portfolio_return'].sum().reset_index()
monthly_pivot = monthly_returns.pivot(index='year', columns='month', values='portfolio_return') * 100

fig, ax = plt.subplots(figsize=(12, 6))
sns.heatmap(monthly_pivot, annot=True, fmt='.1f', cmap='RdYlGn', center=0, 
            cbar_kws={'label': 'Monthly Return (%)'}, ax=ax)
ax.set_title(f'Monthly Returns Heatmap - {first_model}', fontsize=16, fontweight='bold')
ax.set_xlabel('Month')
ax.set_ylabel('Year')
plt.tight_layout()
plt.savefig(output_dir / 'monthly_returns_heatmap.png', dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: {output_dir / 'monthly_returns_heatmap.png'}")
plt.close()

print(f"\n" + "="*80)
print("✅ BACKTESTING COMPLETE!")
print(f"📁 Scorecard: {output_file}")
print(f"📊 Plots: {output_dir}/")
print("="*80 + "\n")
