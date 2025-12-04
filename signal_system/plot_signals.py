"""
Signal Visualization
Generate plots for trading signals and backtest results.

Usage:
    python plot_signals.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Style
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = ['#2563eb', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6']


def plot_cumulative_pnl(backtest_df: pd.DataFrame, output_path: Path):
    """Plot cumulative PnL for all commodities."""
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    for i, commodity in enumerate(backtest_df['commodity'].unique()):
        comm_df = backtest_df[backtest_df['commodity'] == commodity].sort_values('week')
        ax.plot(comm_df['week'], comm_df['cumulative_pnl'] * 100, 
                label=commodity, color=COLORS[i % len(COLORS)], linewidth=2)
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Cumulative PnL (%)', fontsize=12)
    ax.set_title('Cumulative PnL by Commodity', fontsize=14)
    ax.legend(loc='upper left')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    fig.savefig(output_path / 'cumulative_pnl.png', dpi=150)
    plt.close()
    print(f"  Saved: cumulative_pnl.png")


def plot_position_history(signals_df: pd.DataFrame, output_path: Path):
    """Plot position history for all commodities."""
    
    n_commodities = len(signals_df['commodity'].unique())
    fig, axes = plt.subplots(n_commodities, 1, figsize=(14, 3*n_commodities), sharex=True)
    
    if n_commodities == 1:
        axes = [axes]
    
    for i, commodity in enumerate(signals_df['commodity'].unique()):
        comm_df = signals_df[signals_df['commodity'] == commodity].sort_values('week')
        
        ax = axes[i]
        
        # Plot positions as bars
        colors = ['#10b981' if p > 0 else '#ef4444' if p < 0 else '#6b7280' 
                  for p in comm_df['position']]
        ax.bar(comm_df['week'], comm_df['position'], color=colors, alpha=0.7, width=5)
        
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_ylabel('Position', fontsize=10)
        ax.set_title(f'{commodity}', fontsize=11, fontweight='bold')
        ax.set_ylim(-1.2, 1.2)
    
    axes[-1].set_xlabel('Date', fontsize=12)
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)
    
    plt.suptitle('Position History by Commodity', fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(output_path / 'position_history.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: position_history.png")


def plot_weekly_returns(backtest_df: pd.DataFrame, output_path: Path):
    """Plot weekly PnL distribution."""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    
    commodities = backtest_df['commodity'].unique()
    
    for i, commodity in enumerate(commodities):
        if i >= len(axes):
            break
            
        comm_df = backtest_df[backtest_df['commodity'] == commodity]
        returns = comm_df['net_pnl'].values * 100
        
        ax = axes[i]
        
        # Histogram
        ax.hist(returns, bins=30, color=COLORS[i % len(COLORS)], 
                alpha=0.7, edgecolor='white')
        ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
        ax.axvline(x=returns.mean(), color='red', linestyle='-', linewidth=2, 
                   label=f'Mean: {returns.mean():.2f}%')
        
        ax.set_xlabel('Weekly PnL (%)')
        ax.set_ylabel('Frequency')
        ax.set_title(f'{commodity}', fontweight='bold')
        ax.legend(fontsize=8)
    
    # Hide unused axes
    for j in range(len(commodities), len(axes)):
        axes[j].set_visible(False)
    
    plt.suptitle('Weekly PnL Distribution', fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(output_path / 'weekly_returns_dist.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: weekly_returns_dist.png")


def plot_signal_accuracy(backtest_df: pd.DataFrame, output_path: Path):
    """Plot signal accuracy (when positioned, was direction correct?)."""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    commodities = []
    hit_rates = []
    win_rates = []
    
    for commodity in backtest_df['commodity'].unique():
        comm_df = backtest_df[backtest_df['commodity'] == commodity]
        
        # Filter to non-zero positions
        active = comm_df[comm_df['position'] != 0]
        
        if len(active) > 0:
            # Hit ratio: position direction matches return direction
            correct = ((active['position'] > 0) & (active['actual_return'] > 0)) | \
                     ((active['position'] < 0) & (active['actual_return'] < 0))
            hit_ratio = correct.mean()
            
            # Win rate: positive PnL
            win_rate = (active['net_pnl'] > 0).mean()
            
            commodities.append(commodity)
            hit_rates.append(hit_ratio * 100)
            win_rates.append(win_rate * 100)
    
    x = np.arange(len(commodities))
    width = 0.35
    
    ax.bar(x - width/2, hit_rates, width, label='Direction Accuracy', color=COLORS[0])
    ax.bar(x + width/2, win_rates, width, label='Win Rate', color=COLORS[1])
    
    ax.axhline(y=50, color='red', linestyle='--', linewidth=1, label='Random (50%)')
    
    ax.set_ylabel('Percentage (%)')
    ax.set_title('Signal Accuracy by Commodity')
    ax.set_xticks(x)
    ax.set_xticklabels(commodities)
    ax.legend()
    ax.set_ylim(0, 100)
    
    # Add value labels
    for i, (hr, wr) in enumerate(zip(hit_rates, win_rates)):
        ax.text(i - width/2, hr + 2, f'{hr:.1f}%', ha='center', fontsize=9)
        ax.text(i + width/2, wr + 2, f'{wr:.1f}%', ha='center', fontsize=9)
    
    plt.tight_layout()
    fig.savefig(output_path / 'signal_accuracy.png', dpi=150)
    plt.close()
    print(f"  Saved: signal_accuracy.png")


def plot_drawdown(backtest_df: pd.DataFrame, output_path: Path):
    """Plot drawdown for each commodity."""
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    for i, commodity in enumerate(backtest_df['commodity'].unique()):
        comm_df = backtest_df[backtest_df['commodity'] == commodity].sort_values('week')
        
        cumulative = comm_df['cumulative_pnl'].values
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) * 100
        
        ax.fill_between(comm_df['week'], drawdown, 0, 
                        alpha=0.3, color=COLORS[i % len(COLORS)], label=commodity)
        ax.plot(comm_df['week'], drawdown, color=COLORS[i % len(COLORS)], linewidth=1)
    
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Drawdown (%)', fontsize=12)
    ax.set_title('Strategy Drawdown by Commodity', fontsize=14)
    ax.legend(loc='lower left')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    fig.savefig(output_path / 'drawdown.png', dpi=150)
    plt.close()
    print(f"  Saved: drawdown.png")


def plot_monthly_returns_heatmap(backtest_df: pd.DataFrame, output_path: Path):
    """Plot monthly returns heatmap."""
    
    # Aggregate to monthly
    backtest_df = backtest_df.copy()
    backtest_df['month'] = backtest_df['week'].dt.to_period('M')
    
    monthly = backtest_df.groupby(['commodity', 'month'])['net_pnl'].sum().unstack(level=0)
    monthly = monthly * 100  # Convert to percentage
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    im = ax.imshow(monthly.values, cmap='RdYlGn', aspect='auto', vmin=-10, vmax=10)
    
    # Labels
    ax.set_xticks(np.arange(len(monthly.columns)))
    ax.set_yticks(np.arange(len(monthly.index)))
    ax.set_xticklabels(monthly.columns)
    ax.set_yticklabels([str(m) for m in monthly.index])
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add values
    for i in range(len(monthly.index)):
        for j in range(len(monthly.columns)):
            val = monthly.iloc[i, j]
            if not np.isnan(val):
                text = ax.text(j, i, f'{val:.1f}', ha="center", va="center",
                              color="black" if abs(val) < 5 else "white", fontsize=8)
    
    ax.set_title('Monthly Returns by Commodity (%)', fontsize=14)
    fig.colorbar(im, ax=ax, label='Return (%)')
    
    plt.tight_layout()
    fig.savefig(output_path / 'monthly_returns_heatmap.png', dpi=150)
    plt.close()
    print(f"  Saved: monthly_returns_heatmap.png")


def main():
    """Generate all signal visualizations."""
    
    print("="*60)
    print("📊 GENERATING SIGNAL VISUALIZATIONS")
    print("="*60)
    
    # Load data
    signals_path = Path('reports/signals/weekly_signals.csv')
    backtest_path = Path('reports/signals/backtest_results.csv')
    
    if not signals_path.exists() or not backtest_path.exists():
        print("\n❌ Signal files not found. Run signal_generator.py first.")
        return
    
    signals_df = pd.read_csv(signals_path)
    signals_df['week'] = pd.to_datetime(signals_df['week'])
    
    backtest_df = pd.read_csv(backtest_path)
    backtest_df['week'] = pd.to_datetime(backtest_df['week'])
    
    print(f"\n  Loaded {len(signals_df)} signal records")
    print(f"  Loaded {len(backtest_df)} backtest records")
    
    # Output directory
    output_path = Path('reports/signals/plots')
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📈 Generating plots...")
    
    # Generate plots
    plot_cumulative_pnl(backtest_df, output_path)
    plot_position_history(signals_df, output_path)
    plot_weekly_returns(backtest_df, output_path)
    plot_signal_accuracy(backtest_df, output_path)
    plot_drawdown(backtest_df, output_path)
    
    try:
        plot_monthly_returns_heatmap(backtest_df, output_path)
    except Exception as e:
        print(f"  ⚠️ Heatmap skipped: {e}")
    
    print(f"\n✅ All plots saved to: {output_path}/")


if __name__ == '__main__':
    main()
