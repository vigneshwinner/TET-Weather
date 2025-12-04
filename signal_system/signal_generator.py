"""
Trading Signal Generator
Turn model outputs into executable weekly signals with basic risk controls.

Features:
- Magnitude policy: long/short/flat based on predicted return threshold
- Probability policy: long/short based on direction probability cutoffs
- Position sizing by confidence (linear scaling)
- Transaction cost model (bps) with optional slippage
- Max gross exposure per commodity
- Max turnover per week
- Full backtest with PnL calculation

Usage:
    python signal_generator.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class SignalConfig:
    """Configuration for signal generation."""
    
    # Input/Output
    predictions_file: str = 'reports/oos_predictions.csv'
    output_dir: str = 'reports/signals'
    
    # Signal Policy Selection ('magnitude' or 'probability')
    policy: str = 'probability'
    
    # Magnitude Policy Thresholds
    mag_long_threshold: float = 0.005    # Go long if pred_ret > this
    mag_short_threshold: float = -0.005  # Go short if pred_ret < this
    
    # Probability Policy Cutoffs
    prob_long_cutoff: float = 0.55       # Go long if prob > this
    prob_short_cutoff: float = 0.45      # Go short if prob < this
    
    # Position Sizing
    max_position: float = 1.0            # Maximum position size (1.0 = 100%)
    min_position: float = 0.0            # Minimum position size
    use_confidence_scaling: bool = True  # Scale position by confidence
    
    # Transaction Costs
    transaction_cost_bps: float = 5.0    # Cost in basis points (5 bps = 0.05%)
    slippage_bps: float = 2.0            # Additional slippage
    
    # Risk Limits
    max_gross_exposure: float = 1.0      # Max exposure per commodity
    max_turnover_per_week: float = 2.0   # Max turnover (2.0 = 200% of portfolio)
    
    # Backtest Settings
    initial_capital: float = 100000.0    # Starting capital for backtest


# ============================================================================
# Signal Generation
# ============================================================================

class SignalGenerator:
    """Generate trading signals from model predictions."""
    
    def __init__(self, config: SignalConfig = None):
        self.config = config or SignalConfig()
    
    def load_predictions(self) -> pd.DataFrame:
        """Load and prepare predictions data."""
        print(f"📂 Loading predictions from: {self.config.predictions_file}")
        
        df = pd.read_csv(self.config.predictions_file)
        
        # Handle duplicate column names
        if df.columns.duplicated().any():
            cols = df.columns.tolist()
            new_cols = []
            seen = {}
            for col in cols:
                if col in seen:
                    seen[col] += 1
                    new_cols.append(f"{col}_{seen[col]}")
                else:
                    seen[col] = 0
                    new_cols.append(col)
            df.columns = new_cols
        
        # Find the week column
        week_col = None
        for col in ['week_start', 'week_start_1', 'week']:
            if col in df.columns:
                week_col = col
                break
        
        if week_col is None:
            raise ValueError("Could not find week column")
        
        # Rename to standard
        df = df.rename(columns={week_col: 'week'})
        df['week'] = pd.to_datetime(df['week'])
        
        print(f"  Loaded {len(df)} rows")
        print(f"  Date range: {df['week'].min().date()} to {df['week'].max().date()}")
        print(f"  Commodities: {df['commodity'].unique().tolist()}")
        
        return df
    
    def aggregate_to_weekly(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate daily predictions to weekly signals."""
        
        # Group by commodity and week, take first prediction (they're the same per week)
        weekly = df.groupby(['commodity', 'week']).agg({
            'y_pred_ret': 'first',
            'y_pred_dir_prob': 'first',
            'y_true': 'sum',  # Sum daily returns for weekly return
            'model_name': 'first'
        }).reset_index()
        
        weekly = weekly.sort_values(['commodity', 'week']).reset_index(drop=True)
        
        print(f"\n📊 Aggregated to {len(weekly)} weekly observations")
        
        return weekly
    
    def generate_magnitude_signal(self, pred_ret: float) -> int:
        """
        Generate signal based on predicted return magnitude.
        
        Returns:
            1 = Long, -1 = Short, 0 = Flat
        """
        if pred_ret > self.config.mag_long_threshold:
            return 1
        elif pred_ret < self.config.mag_short_threshold:
            return -1
        else:
            return 0
    
    def generate_probability_signal(self, prob: float) -> int:
        """
        Generate signal based on direction probability.
        
        Returns:
            1 = Long, -1 = Short, 0 = Flat
        """
        if prob > self.config.prob_long_cutoff:
            return 1
        elif prob < self.config.prob_short_cutoff:
            return -1
        else:
            return 0
    
    def calculate_position_size(self, pred_ret: float, prob: float, signal: int) -> float:
        """
        Calculate position size based on confidence.
        
        Uses linear scaling from min to max position based on:
        - Magnitude policy: absolute predicted return
        - Probability policy: distance from 0.5
        """
        if signal == 0:
            return 0.0
        
        if not self.config.use_confidence_scaling:
            return self.config.max_position * abs(signal)
        
        if self.config.policy == 'magnitude':
            # Scale by absolute predicted return
            # Normalize: assume max expected return is ~5%
            confidence = min(abs(pred_ret) / 0.05, 1.0)
        else:
            # Scale by probability distance from 0.5
            confidence = abs(prob - 0.5) * 2  # Range: 0 to 1
        
        # Linear interpolation between min and max position
        position = self.config.min_position + confidence * (
            self.config.max_position - self.config.min_position
        )
        
        return min(position, self.config.max_position)
    
    def apply_risk_limits(
        self, 
        position: float, 
        prev_position: float
    ) -> Tuple[float, float]:
        """
        Apply risk limits to position.
        
        Returns:
            (adjusted_position, turnover)
        """
        # Enforce max gross exposure
        position = np.clip(position, -self.config.max_gross_exposure, 
                          self.config.max_gross_exposure)
        
        # Calculate turnover
        turnover = abs(position - prev_position)
        
        # Enforce max turnover
        if turnover > self.config.max_turnover_per_week:
            # Scale down the change
            max_change = self.config.max_turnover_per_week
            if position > prev_position:
                position = prev_position + max_change
            else:
                position = prev_position - max_change
            turnover = max_change
        
        return position, turnover
    
    def calculate_transaction_cost(self, turnover: float) -> float:
        """Calculate transaction cost in decimal form."""
        total_bps = self.config.transaction_cost_bps + self.config.slippage_bps
        return turnover * (total_bps / 10000)
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate signals for all commodities and weeks."""
        
        print(f"\n🎯 Generating signals using {self.config.policy.upper()} policy...")
        
        results = []
        
        for commodity in df['commodity'].unique():
            comm_df = df[df['commodity'] == commodity].sort_values('week')
            
            prev_position = 0.0
            
            for _, row in comm_df.iterrows():
                # Generate raw signal
                if self.config.policy == 'magnitude':
                    raw_signal = self.generate_magnitude_signal(row['y_pred_ret'])
                else:
                    raw_signal = self.generate_probability_signal(row['y_pred_dir_prob'])
                
                # Calculate position size
                raw_position = self.calculate_position_size(
                    row['y_pred_ret'], 
                    row['y_pred_dir_prob'],
                    raw_signal
                ) * raw_signal
                
                # Apply risk limits
                position, turnover = self.apply_risk_limits(raw_position, prev_position)
                
                # Calculate transaction cost
                tc = self.calculate_transaction_cost(turnover)
                
                results.append({
                    'commodity': commodity,
                    'week': row['week'],
                    'y_pred_ret': row['y_pred_ret'],
                    'y_pred_dir_prob': row['y_pred_dir_prob'],
                    'y_true': row['y_true'],
                    'raw_signal': raw_signal,
                    'raw_position': raw_position,
                    'position': position,
                    'turnover': turnover,
                    'transaction_cost': tc,
                    'prev_position': prev_position
                })
                
                prev_position = position
        
        signals_df = pd.DataFrame(results)
        
        # Print summary
        print(f"\n  Signal Distribution:")
        for commodity in signals_df['commodity'].unique():
            comm_signals = signals_df[signals_df['commodity'] == commodity]
            long_pct = (comm_signals['raw_signal'] == 1).mean() * 100
            short_pct = (comm_signals['raw_signal'] == -1).mean() * 100
            flat_pct = (comm_signals['raw_signal'] == 0).mean() * 100
            print(f"    {commodity}: Long {long_pct:.1f}% | Short {short_pct:.1f}% | Flat {flat_pct:.1f}%")
        
        return signals_df
    
    def run_backtest(self, signals_df: pd.DataFrame) -> pd.DataFrame:
        """Run backtest on generated signals."""
        
        print(f"\n📈 Running backtest...")
        
        results = []
        
        for commodity in signals_df['commodity'].unique():
            comm_df = signals_df[signals_df['commodity'] == commodity].sort_values('week')
            
            for _, row in comm_df.iterrows():
                # PnL = position * return - transaction cost
                gross_pnl = row['position'] * row['y_true']
                net_pnl = gross_pnl - row['transaction_cost']
                
                results.append({
                    'commodity': commodity,
                    'week': row['week'],
                    'position': row['position'],
                    'actual_return': row['y_true'],
                    'gross_pnl': gross_pnl,
                    'transaction_cost': row['transaction_cost'],
                    'net_pnl': net_pnl,
                    'turnover': row['turnover'],
                    'y_pred_ret': row['y_pred_ret'],
                    'y_pred_dir_prob': row['y_pred_dir_prob'],
                    'signal': row['raw_signal']
                })
        
        backtest_df = pd.DataFrame(results)
        
        # Calculate cumulative PnL
        backtest_df['cumulative_pnl'] = backtest_df.groupby('commodity')['net_pnl'].cumsum()
        
        return backtest_df
    
    def calculate_performance_metrics(self, backtest_df: pd.DataFrame) -> Dict:
        """Calculate performance metrics from backtest results."""
        
        metrics = {}
        
        for commodity in backtest_df['commodity'].unique():
            comm_df = backtest_df[backtest_df['commodity'] == commodity]
            
            returns = comm_df['net_pnl'].values
            
            # Basic metrics
            total_return = returns.sum()
            avg_return = returns.mean()
            std_return = returns.std()
            
            # Sharpe ratio (annualized, assuming weekly data)
            sharpe = (avg_return / std_return * np.sqrt(52)) if std_return > 0 else 0
            
            # Win rate
            win_rate = (returns > 0).mean()
            
            # Max drawdown
            cumulative = np.cumsum(returns)
            running_max = np.maximum.accumulate(cumulative)
            drawdown = running_max - cumulative
            max_drawdown = drawdown.max()
            
            # Hit ratio (correct direction)
            positions = comm_df['position'].values
            actual_returns = comm_df['actual_return'].values
            correct_direction = ((positions > 0) & (actual_returns > 0)) | \
                              ((positions < 0) & (actual_returns < 0)) | \
                              (positions == 0)
            hit_ratio = correct_direction.mean()
            
            # Profit factor
            gross_profits = returns[returns > 0].sum()
            gross_losses = abs(returns[returns < 0].sum())
            profit_factor = gross_profits / gross_losses if gross_losses > 0 else np.inf
            
            # Total transaction costs
            total_tc = comm_df['transaction_cost'].sum()
            
            # Average turnover
            avg_turnover = comm_df['turnover'].mean()
            
            metrics[commodity] = {
                'total_return': total_return,
                'avg_weekly_return': avg_return,
                'std_weekly_return': std_return,
                'sharpe_ratio': sharpe,
                'win_rate': win_rate,
                'hit_ratio': hit_ratio,
                'max_drawdown': max_drawdown,
                'profit_factor': profit_factor,
                'total_transaction_costs': total_tc,
                'avg_turnover': avg_turnover,
                'n_weeks': len(comm_df)
            }
        
        return metrics
    
    def print_performance_summary(self, metrics: Dict):
        """Print formatted performance summary."""
        
        print("\n" + "="*80)
        print("📊 BACKTEST PERFORMANCE SUMMARY")
        print("="*80)
        
        print(f"\n{'Commodity':<12} {'Total Ret':>10} {'Sharpe':>8} {'Win Rate':>10} {'Hit Ratio':>10} {'Max DD':>10} {'Profit F':>10}")
        print("-"*80)
        
        for commodity, m in metrics.items():
            print(f"{commodity:<12} {m['total_return']*100:>9.2f}% {m['sharpe_ratio']:>8.2f} "
                  f"{m['win_rate']*100:>9.1f}% {m['hit_ratio']*100:>9.1f}% "
                  f"{m['max_drawdown']*100:>9.2f}% {m['profit_factor']:>10.2f}")
        
        # Overall
        print("-"*80)
        all_returns = [m['total_return'] for m in metrics.values()]
        all_sharpes = [m['sharpe_ratio'] for m in metrics.values()]
        all_win_rates = [m['win_rate'] for m in metrics.values()]
        
        print(f"{'AVERAGE':<12} {np.mean(all_returns)*100:>9.2f}% {np.mean(all_sharpes):>8.2f} "
              f"{np.mean(all_win_rates)*100:>9.1f}%")
        
        print("\n📋 Configuration:")
        print(f"  Policy: {self.config.policy}")
        if self.config.policy == 'magnitude':
            print(f"  Long threshold: {self.config.mag_long_threshold}")
            print(f"  Short threshold: {self.config.mag_short_threshold}")
        else:
            print(f"  Long cutoff: {self.config.prob_long_cutoff}")
            print(f"  Short cutoff: {self.config.prob_short_cutoff}")
        print(f"  Transaction cost: {self.config.transaction_cost_bps} bps")
        print(f"  Slippage: {self.config.slippage_bps} bps")
        print(f"  Max position: {self.config.max_position}")
        print(f"  Confidence scaling: {self.config.use_confidence_scaling}")
    
    def save_results(self, signals_df: pd.DataFrame, backtest_df: pd.DataFrame, metrics: Dict):
        """Save all results to files."""
        
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save signals
        signals_path = output_dir / 'weekly_signals.csv'
        signals_df.to_csv(signals_path, index=False)
        
        # Save backtest results
        backtest_path = output_dir / 'backtest_results.csv'
        backtest_df.to_csv(backtest_path, index=False)
        
        # Save metrics
        metrics_df = pd.DataFrame(metrics).T
        metrics_df.index.name = 'commodity'
        metrics_path = output_dir / 'performance_metrics.csv'
        metrics_df.to_csv(metrics_path)
        
        # Save configuration
        config_dict = {
            'policy': self.config.policy,
            'mag_long_threshold': self.config.mag_long_threshold,
            'mag_short_threshold': self.config.mag_short_threshold,
            'prob_long_cutoff': self.config.prob_long_cutoff,
            'prob_short_cutoff': self.config.prob_short_cutoff,
            'max_position': self.config.max_position,
            'transaction_cost_bps': self.config.transaction_cost_bps,
            'slippage_bps': self.config.slippage_bps,
            'max_gross_exposure': self.config.max_gross_exposure,
            'max_turnover_per_week': self.config.max_turnover_per_week,
            'use_confidence_scaling': self.config.use_confidence_scaling
        }
        
        import json
        config_path = output_dir / 'signal_config.json'
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        print(f"\n💾 Results saved to: {output_dir}/")
        print(f"  - weekly_signals.csv")
        print(f"  - backtest_results.csv")
        print(f"  - performance_metrics.csv")
        print(f"  - signal_config.json")
        
        return {
            'signals': signals_path,
            'backtest': backtest_path,
            'metrics': metrics_path,
            'config': config_path
        }
    
    def run(self) -> Dict:
        """Run the full signal generation pipeline."""
        
        print("="*80)
        print("🚀 TRADING SIGNAL GENERATOR")
        print("="*80)
        
        # 1. Load predictions
        df = self.load_predictions()
        
        # 2. Aggregate to weekly
        weekly_df = self.aggregate_to_weekly(df)
        
        # 3. Generate signals
        signals_df = self.generate_signals(weekly_df)
        
        # 4. Run backtest
        backtest_df = self.run_backtest(signals_df)
        
        # 5. Calculate metrics
        metrics = self.calculate_performance_metrics(backtest_df)
        
        # 6. Print summary
        self.print_performance_summary(metrics)
        
        # 7. Save results
        saved_files = self.save_results(signals_df, backtest_df, metrics)
        
        print("\n" + "="*80)
        print("✅ SIGNAL GENERATION COMPLETE!")
        print("="*80)
        
        return {
            'signals': signals_df,
            'backtest': backtest_df,
            'metrics': metrics,
            'saved_files': saved_files
        }


# ============================================================================
# Main
# ============================================================================

def main():
    """Run signal generator with default configuration."""
    
    # Create configuration
    config = SignalConfig(
        # Input
        predictions_file='reports/oos_predictions.csv',
        output_dir='reports/signals',
        
        # Policy: 'magnitude' or 'probability'
        policy='probability',
        
        # Magnitude thresholds (if using magnitude policy)
        mag_long_threshold=0.005,   # 0.5%
        mag_short_threshold=-0.005, # -0.5%
        
        # Probability cutoffs (if using probability policy)
        prob_long_cutoff=0.55,      # Go long if prob > 55%
        prob_short_cutoff=0.45,     # Go short if prob < 45%
        
        # Position sizing
        max_position=1.0,
        use_confidence_scaling=True,
        
        # Transaction costs
        transaction_cost_bps=5.0,
        slippage_bps=2.0,
        
        # Risk limits
        max_gross_exposure=1.0,
        max_turnover_per_week=2.0
    )
    
    # Run generator
    generator = SignalGenerator(config)
    results = generator.run()
    
    return results


if __name__ == '__main__':
    main()
