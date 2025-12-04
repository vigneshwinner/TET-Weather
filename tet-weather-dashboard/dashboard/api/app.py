"""
TET-Weather Forecast Dashboard API
Flask backend serving predictions, signals, and performance data.

Run: python app.py
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json

app = Flask(__name__)
CORS(app)

# ============================================================================
# Configuration
# ============================================================================

BASE_DIR = Path(__file__).parent.parent  # Assumes api/ is in dashboard/
DATA_DIR = BASE_DIR.parent.parent  # Go up to TET-Weather root (tet-weather-dashboard/dashboard/ -> tet-weather-dashboard/ -> TET-Weather/)

# Data file paths (relative to TET-Weather root)
PREDICTIONS_FILE = DATA_DIR / 'reports' / 'oos_predictions.csv'
SIGNALS_FILE = DATA_DIR / 'reports' / 'signals' / 'weekly_signals.csv'
BACKTEST_FILE = DATA_DIR / 'reports' / 'signals' / 'backtest_results.csv'
METRICS_FILE = DATA_DIR / 'reports' / 'signals' / 'performance_metrics.csv'
EVAL_FILE = DATA_DIR / 'reports' / 'evaluation_results.csv'

# Cache for loaded data
_cache = {}


# ============================================================================
# Data Loading
# ============================================================================

def load_data(file_path, cache_key):
    """Load CSV with caching."""
    if cache_key not in _cache:
        if file_path.exists():
            _cache[cache_key] = pd.read_csv(file_path)
        else:
            _cache[cache_key] = None
    return _cache[cache_key]


def clear_cache():
    """Clear data cache."""
    global _cache
    _cache = {}


# ============================================================================
# API Routes
# ============================================================================

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'data_files': {
            'predictions': PREDICTIONS_FILE.exists(),
            'signals': SIGNALS_FILE.exists(),
            'backtest': BACKTEST_FILE.exists(),
            'metrics': METRICS_FILE.exists()
        }
    })


@app.route('/api/commodities', methods=['GET'])
def get_commodities():
    """Get list of available commodities."""
    df = load_data(PREDICTIONS_FILE, 'predictions')
    
    if df is None:
        return jsonify({'commodities': ['Brent', 'Henry_Hub', 'Power', 'Copper', 'Corn']})
    
    commodities = df['commodity'].unique().tolist()
    return jsonify({'commodities': commodities})


@app.route('/api/forecast', methods=['GET'])
def get_forecast():
    """Get latest forecast for a commodity."""
    commodity = request.args.get('commodity', 'Henry_Hub')
    
    df = load_data(PREDICTIONS_FILE, 'predictions')
    
    if df is None:
        # Return mock data
        return jsonify({
            'commodity': commodity,
            'date': datetime.now().strftime('%Y-%m-%d'),
            'predicted_return': 0.005,
            'direction_probability': 0.62,
            'signal': 'LONG',
            'confidence': 0.62
        })
    
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
    
    # Get week column
    week_col = 'week_start' if 'week_start' in df.columns else 'week_start_1'
    
    # Filter to commodity and get latest
    comm_df = df[df['commodity'] == commodity].copy()
    comm_df[week_col] = pd.to_datetime(comm_df[week_col])
    latest = comm_df.sort_values(week_col).iloc[-1]
    
    # Determine signal
    prob = latest['y_pred_dir_prob']
    if prob > 0.55:
        signal = 'LONG'
    elif prob < 0.45:
        signal = 'SHORT'
    else:
        signal = 'FLAT'
    
    # Get current price from price file
    current_price = None
    target_price = None
    try:
        price_file = DATA_DIR / 'cleaned_data' / f'{commodity}_3yr.csv'
        if price_file.exists():
            # Different commodities have different header formats
            if commodity in ['Corn', 'Power']:
                # Corn and Power have 2 header rows
                price_df = pd.read_csv(price_file, skiprows=2)
            else:
                # Brent, Henry_Hub, Copper have 3 header rows (includes empty Date,,,,, line)
                price_df = pd.read_csv(price_file, skiprows=3)
                price_df.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']
            
            current_price = float(price_df['Close'].iloc[-1])
            
            # Corn is in cents per bushel, convert to dollars
            if commodity == 'Corn':
                current_price = current_price / 100.0
            
            predicted_return = float(latest['y_pred_ret'])
            target_price = current_price * (1 + predicted_return)
    except Exception as e:
        print(f"Error loading price data for {commodity}: {e}")
    
    return jsonify({
        'commodity': commodity,
        'date': str(latest[week_col].date()),
        'predicted_return': round(float(latest['y_pred_ret']), 6),
        'direction_probability': round(float(prob), 4),
        'signal': signal,
        'confidence': round(abs(prob - 0.5) * 2, 4),
        'current_price': round(current_price, 2) if current_price else None,
        'target_price': round(target_price, 2) if target_price else None
    })


@app.route('/api/signals', methods=['GET'])
def get_signals():
    """Get signal history for a commodity."""
    commodity = request.args.get('commodity', 'Henry_Hub')
    limit = int(request.args.get('limit', 52))
    
    df = load_data(SIGNALS_FILE, 'signals')
    
    if df is None:
        return jsonify({'signals': [], 'error': 'Signals file not found'})
    
    comm_df = df[df['commodity'] == commodity].copy()
    comm_df['week'] = pd.to_datetime(comm_df['week'])
    comm_df = comm_df.sort_values('week').tail(limit)
    
    signals = []
    for _, row in comm_df.iterrows():
        signals.append({
            'date': str(row['week'].date()),
            'signal': int(row['raw_signal']),
            'position': round(float(row['position']), 4),
            'predicted_return': round(float(row['y_pred_ret']), 6),
            'probability': round(float(row['y_pred_dir_prob']), 4)
        })
    
    return jsonify({'commodity': commodity, 'signals': signals})


@app.route('/api/backtest', methods=['GET'])
def get_backtest():
    """Get backtest results for a commodity."""
    commodity = request.args.get('commodity', 'Henry_Hub')
    
    df = load_data(BACKTEST_FILE, 'backtest')
    
    if df is None:
        return jsonify({'backtest': [], 'error': 'Backtest file not found'})
    
    comm_df = df[df['commodity'] == commodity].copy()
    comm_df['week'] = pd.to_datetime(comm_df['week'])
    comm_df = comm_df.sort_values('week')
    
    results = []
    for _, row in comm_df.iterrows():
        results.append({
            'date': str(row['week'].date()),
            'position': round(float(row['position']), 4),
            'actual_return': round(float(row['actual_return']), 6),
            'pnl': round(float(row['net_pnl']), 6),
            'cumulative_pnl': round(float(row['cumulative_pnl']), 6)
        })
    
    return jsonify({'commodity': commodity, 'backtest': results})


@app.route('/api/performance', methods=['GET'])
def get_performance():
    """Get performance metrics for all commodities."""
    df = load_data(METRICS_FILE, 'metrics')
    
    if df is None:
        # Return mock data
        return jsonify({
            'metrics': {
                'Brent': {'total_return': -0.08, 'sharpe_ratio': -0.29, 'win_rate': 0.32, 'max_drawdown': 0.16},
                'Henry_Hub': {'total_return': -0.19, 'sharpe_ratio': -0.22, 'win_rate': 0.37, 'max_drawdown': 0.30},
                'Power': {'total_return': 0.47, 'sharpe_ratio': 0.05, 'win_rate': 0.26, 'max_drawdown': 1.16},
                'Copper': {'total_return': -0.23, 'sharpe_ratio': -0.62, 'win_rate': 0.36, 'max_drawdown': 0.25},
                'Corn': {'total_return': -0.11, 'sharpe_ratio': -0.44, 'win_rate': 0.31, 'max_drawdown': 0.15}
            }
        })
    
    metrics = {}
    for _, row in df.iterrows():
        commodity = row.get('commodity', row.name)
        metrics[commodity] = {
            'total_return': round(float(row.get('total_return', 0)), 4),
            'sharpe_ratio': round(float(row.get('sharpe_ratio', 0)), 4),
            'win_rate': round(float(row.get('win_rate', 0)), 4),
            'hit_ratio': round(float(row.get('hit_ratio', 0)), 4),
            'max_drawdown': round(float(row.get('max_drawdown', 0)), 4),
            'profit_factor': round(float(row.get('profit_factor', 0)), 4),
            'avg_turnover': round(float(row.get('avg_turnover', 0)), 4)
        }
    
    return jsonify({'metrics': metrics})


@app.route('/api/ssi', methods=['GET'])
def get_ssi():
    """Get SSI (Signal Strength Index) time series for a commodity."""
    commodity = request.args.get('commodity', 'Henry_Hub')
    start_date = request.args.get('start', None)
    end_date = request.args.get('end', None)
    
    signals_df = load_data(SIGNALS_FILE, 'signals')
    
    if signals_df is None:
        return jsonify({'ssi': [], 'error': 'Signals file not found'})
    
    # Filter by commodity
    comm_df = signals_df[signals_df['commodity'] == commodity].copy()
    comm_df['week'] = pd.to_datetime(comm_df['week'])
    
    # Apply date filters
    if start_date:
        comm_df = comm_df[comm_df['week'] >= pd.to_datetime(start_date)]
    if end_date:
        comm_df = comm_df[comm_df['week'] <= pd.to_datetime(end_date)]
    
    comm_df = comm_df.sort_values('week')
    
    # Calculate SSI (use probability as signal strength)
    ssi_data = []
    for _, row in comm_df.iterrows():
        # SSI = probability weighted by position size
        prob = float(row.get('y_pred_dir_prob', 0.5))
        position = float(row.get('position', 0))
        ssi = (prob - 0.5) * 2 * abs(position)  # Scale to [-1, 1]
        
        ssi_data.append({
            'date': str(row['week'].date()),
            'ssi': round(ssi, 4),
            'probability': round(prob, 4),
            'position': round(position, 4),
            'signal': int(row.get('raw_signal', 0))
        })
    
    return jsonify({
        'commodity': commodity,
        'start_date': start_date,
        'end_date': end_date,
        'ssi': ssi_data
    })


@app.route('/api/backtest/summary', methods=['GET'])
def get_backtest_summary():
    """Get backtest performance summary for a commodity."""
    commodity = request.args.get('commodity', None)
    
    metrics_df = load_data(METRICS_FILE, 'metrics')
    backtest_df = load_data(BACKTEST_FILE, 'backtest')
    
    if metrics_df is None or backtest_df is None:
        return jsonify({'error': 'Data files not found'})
    
    # If no commodity specified, return portfolio summary
    if commodity is None:
        # Aggregate metrics across all commodities
        total_return = metrics_df['total_return'].mean() if 'total_return' in metrics_df.columns else 0
        sharpe = metrics_df['sharpe_ratio'].mean() if 'sharpe_ratio' in metrics_df.columns else 0
        max_dd = metrics_df['max_drawdown'].max() if 'max_drawdown' in metrics_df.columns else 0
        hit_ratio = metrics_df['hit_ratio'].mean() if 'hit_ratio' in metrics_df.columns else 0
        
        # Get recent equity curve points (last 20 weeks)
        backtest_df['week'] = pd.to_datetime(backtest_df['week'])
        recent_equity = backtest_df.groupby('week')['cumulative_pnl'].sum().tail(20)
        equity_points = [
            {'date': str(date.date()), 'pnl': round(float(pnl), 4)}
            for date, pnl in recent_equity.items()
        ]
        
        return jsonify({
            'commodity': 'Portfolio',
            'metrics': {
                'total_return': round(float(total_return), 4),
                'sharpe_ratio': round(float(sharpe), 4),
                'max_drawdown': round(float(max_dd), 4),
                'hit_ratio': round(float(hit_ratio), 4)
            },
            'recent_equity': equity_points
        })
    
    # Commodity-specific summary
    comm_metrics = metrics_df[metrics_df['commodity'] == commodity]
    comm_backtest = backtest_df[backtest_df['commodity'] == commodity].copy()
    
    if len(comm_metrics) == 0 or len(comm_backtest) == 0:
        return jsonify({'error': f'No data for commodity: {commodity}'})
    
    metrics_row = comm_metrics.iloc[0]
    comm_backtest['week'] = pd.to_datetime(comm_backtest['week'])
    comm_backtest = comm_backtest.sort_values('week')
    
    # Get recent equity curve (last 20 weeks)
    recent_equity = comm_backtest.tail(20)
    equity_points = [
        {'date': str(row['week'].date()), 'pnl': round(float(row['cumulative_pnl']), 4)}
        for _, row in recent_equity.iterrows()
    ]
    
    return jsonify({
        'commodity': commodity,
        'metrics': {
            'total_return': round(float(metrics_row.get('total_return', 0)), 4),
            'sharpe_ratio': round(float(metrics_row.get('sharpe_ratio', 0)), 4),
            'sortino_ratio': round(float(metrics_row.get('sortino_ratio', 0)), 4),
            'max_drawdown': round(float(metrics_row.get('max_drawdown', 0)), 4),
            'hit_ratio': round(float(metrics_row.get('hit_ratio', 0)), 4),
            'profit_factor': round(float(metrics_row.get('profit_factor', 0)), 4),
            'avg_turnover': round(float(metrics_row.get('avg_turnover', 0)), 4),
            'total_trades': int(len(comm_backtest))
        },
        'recent_equity': equity_points,
        'start_date': str(comm_backtest.iloc[0]['week'].date()),
        'end_date': str(comm_backtest.iloc[-1]['week'].date())
    })


@app.route('/api/evaluation', methods=['GET'])
def get_evaluation():
    """Get model evaluation metrics."""
    df = load_data(EVAL_FILE, 'evaluation')
    
    if df is None:
        return jsonify({
            'evaluation': {
                'Brent': {'mae': 0.0143, 'r2': -0.06, 'accuracy': 0.50, 'roc_auc': 0.48},
                'Henry_Hub': {'mae': 0.0344, 'r2': -0.08, 'accuracy': 0.50, 'roc_auc': 0.49},
                'Power': {'mae': 0.3984, 'r2': -0.05, 'accuracy': 0.51, 'roc_auc': 0.50},
                'Copper': {'mae': 0.0125, 'r2': -0.08, 'accuracy': 0.49, 'roc_auc': 0.48},
                'Corn': {'mae': 0.0113, 'r2': -0.11, 'accuracy': 0.47, 'roc_auc': 0.47}
            }
        })
    
    evaluation = {}
    for _, row in df.iterrows():
        evaluation[row['commodity']] = {
            'mae': round(float(row['mae']), 4),
            'rmse': round(float(row['rmse']), 4),
            'r2': round(float(row['r2']), 4),
            'accuracy': round(float(row['accuracy']), 4),
            'f1': round(float(row['f1']), 4),
            'roc_auc': round(float(row.get('roc_auc', 0)), 4) if pd.notna(row.get('roc_auc')) else None,
            'brier_score': round(float(row.get('brier_score', 0)), 4) if pd.notna(row.get('brier_score')) else None
        }
    
    return jsonify({'evaluation': evaluation})


@app.route('/api/equity_curve', methods=['GET'])
def get_equity_curve():
    """Get equity curve data for all commodities."""
    df = load_data(BACKTEST_FILE, 'backtest')
    
    if df is None:
        return jsonify({'curves': {}})
    
    curves = {}
    for commodity in df['commodity'].unique():
        comm_df = df[df['commodity'] == commodity].copy()
        comm_df['week'] = pd.to_datetime(comm_df['week'])
        comm_df = comm_df.sort_values('week')
        
        curves[commodity] = {
            'dates': [str(d.date()) for d in comm_df['week']],
            'cumulative_pnl': [round(float(x), 6) for x in comm_df['cumulative_pnl']]
        }
    
    return jsonify({'curves': curves})


@app.route('/api/refresh', methods=['POST'])
def refresh_data():
    """Clear cache and reload data."""
    clear_cache()
    return jsonify({'status': 'cache_cleared', 'timestamp': datetime.now().isoformat()})


@app.route('/api/analysis/price_anomaly', methods=['GET'])
def get_price_anomaly_analysis():
    """Get price vs weather anomaly data for analysis charts."""
    commodity = request.args.get('commodity', 'Brent')
    
    # Load backtest data which has weekly data
    backtest_df = load_data(BACKTEST_FILE, 'backtest')
    if backtest_df is None:
        return jsonify({'error': 'Backtest data not found'})
    
    # Filter by commodity
    comm_bt = backtest_df[backtest_df['commodity'] == commodity].copy()
    comm_bt['week'] = pd.to_datetime(comm_bt['week'])
    comm_bt = comm_bt.sort_values('week')
    
    # Load anomalies (using correct column names)
    anomaly_file = DATA_DIR / 'cleaned_data' / 'nasa_power_weather_daily_with_anomalies_z3.csv'
    if not anomaly_file.exists():
        return jsonify({'error': 'Anomaly data not found'})
    
    anom_df = pd.read_csv(anomaly_file)
    anom_df['date'] = pd.to_datetime(anom_df['date'])
    anom_df['week'] = anom_df['date'] - pd.to_timedelta(anom_df['date'].dt.dayofweek, unit='D')
    
    # Aggregate anomalies by week (using correct column names)
    anom_weekly = anom_df.groupby('week').agg({
        'temp_avg_c_z': 'mean',
        'precipitation_mm_z': 'mean',
        'wind_speed_ms_z': 'mean',
        'temp_avg_c_anomaly_z3': 'max',
        'precipitation_mm_anomaly_z3': 'max'
    }).reset_index()
    
    # Merge
    merged = comm_bt.merge(anom_weekly, on='week', how='left')
    merged = merged.dropna(subset=['temp_avg_c_z'])
    
    # Get actual prices for the commodity
    price_file = DATA_DIR / 'cleaned_data' / f'{commodity}_3yr.csv'
    if price_file.exists():
        price_df = pd.read_csv(price_file, skiprows=2)
        price_df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        price_df['Date'] = pd.to_datetime(price_df['Date'])
        price_df['week'] = price_df['Date'] - pd.to_timedelta(price_df['Date'].dt.dayofweek, unit='D')
        weekly_prices = price_df.groupby('week')['Close'].mean().reset_index()
        merged = merged.merge(weekly_prices, on='week', how='left')
    else:
        merged['Close'] = 100  # Fallback
    
    # Prepare response
    result = {
        'commodity': commodity,
        'dates': [str(d.date()) for d in merged['week']],
        'prices': [round(float(p), 2) if not pd.isna(p) else 0 for p in merged['Close']],
        'returns': [round(float(r * 100), 4) for r in merged['actual_return']],
        'temp_anomaly': [round(float(t), 4) for t in merged['temp_avg_c_z']],
        'precip_anomaly': [round(float(p), 4) for p in merged['precipitation_mm_z']],
        'wind_anomaly': [round(float(w), 4) for w in merged['wind_speed_ms_z']],
        'extreme_temp_dates': [str(merged.iloc[i]['week'].date()) for i in range(len(merged)) if merged.iloc[i]['temp_avg_c_anomaly_z3'] == 1],
        'extreme_precip_dates': [str(merged.iloc[i]['week'].date()) for i in range(len(merged)) if merged.iloc[i]['precipitation_mm_anomaly_z3'] == 1]
    }
    
    return jsonify(result)


@app.route('/api/analysis/anomaly_return_correlation', methods=['GET'])
def get_anomaly_return_correlation():
    """Get correlation between anomalies and returns."""
    commodity = request.args.get('commodity', 'Brent')
    anomaly_type = request.args.get('anomaly_type', 'temp_anomaly')
    
    # Map frontend names to actual column names
    anomaly_map = {
        'temp_anomaly': 'temp_avg_c_z',
        'precip_anomaly': 'precipitation_mm_z',
        'wind_anomaly': 'wind_speed_ms_z'
    }
    actual_col = anomaly_map.get(anomaly_type, 'temp_avg_c_z')
    
    # Load backtest data with actual returns
    backtest_df = load_data(BACKTEST_FILE, 'backtest')
    if backtest_df is None:
        return jsonify({'error': 'Backtest not found'})
    
    # Load anomalies
    anomaly_file = DATA_DIR / 'cleaned_data' / 'nasa_power_weather_daily_with_anomalies_z3.csv'
    if not anomaly_file.exists():
        return jsonify({'error': 'Anomaly data not found'})
    
    anom_df = pd.read_csv(anomaly_file)
    anom_df['date'] = pd.to_datetime(anom_df['date'])
    anom_df['week'] = anom_df['date'] - pd.to_timedelta(anom_df['date'].dt.dayofweek, unit='D')
    
    # Aggregate by week
    anom_weekly = anom_df.groupby('week')[actual_col].mean().reset_index()
    
    # Get commodity backtest data
    comm_bt = backtest_df[backtest_df['commodity'] == commodity].copy()
    comm_bt['week'] = pd.to_datetime(comm_bt['week'])
    
    # Merge
    merged = comm_bt.merge(anom_weekly, on='week', how='inner')
    merged = merged.dropna(subset=[actual_col, 'actual_return'])
    
    if len(merged) == 0:
        return jsonify({'error': 'No data after merge'})
    
    # Calculate bins
    merged['anomaly_bin'] = pd.cut(merged[actual_col], bins=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
    
    # Bin statistics
    bin_stats = merged.groupby('anomaly_bin').agg({
        'actual_return': ['mean', 'std', 'count']
    }).reset_index()
    bin_stats.columns = ['bin', 'mean_return', 'std_return', 'count']
    
    # Regression
    correlation = np.corrcoef(merged[actual_col], merged['actual_return'])[0, 1]
    
    result = {
        'commodity': commodity,
        'anomaly_type': anomaly_type,
        'scatter': {
            'anomaly_values': [round(float(x), 4) for x in merged[actual_col]],
            'returns': [round(float(x * 100), 4) for x in merged['actual_return']],
            'weeks': [str(w.date()) for w in merged['week']]
        },
        'bins': {
            'labels': bin_stats['bin'].tolist(),
            'mean_returns': [round(float(x * 100), 4) for x in bin_stats['mean_return']],
            'std_returns': [round(float(x * 100), 4) for x in bin_stats['std_return']],
            'counts': [int(x) for x in bin_stats['count']]
        },
        'correlation': round(float(correlation), 4),
        'r_squared': round(float(correlation ** 2), 4)
    }
    
    return jsonify(result)


@app.route('/api/analysis/rolling_metrics', methods=['GET'])
def get_rolling_metrics():
    """Get rolling hit ratio and performance metrics."""
    commodity = request.args.get('commodity', 'Brent')
    window = int(request.args.get('window', 20))
    
    backtest_df = load_data(BACKTEST_FILE, 'backtest')
    if backtest_df is None:
        return jsonify({'error': 'Backtest not found'})
    
    # Get commodity data
    comm_bt = backtest_df[backtest_df['commodity'] == commodity].copy()
    comm_bt['week'] = pd.to_datetime(comm_bt['week'])
    comm_bt = comm_bt.sort_values('week')
    
    # Calculate correct predictions (position matched direction of return)
    comm_bt['correct'] = ((comm_bt['position'] > 0) & (comm_bt['actual_return'] > 0)) | ((comm_bt['position'] < 0) & (comm_bt['actual_return'] < 0))
    comm_bt['correct'] = comm_bt['correct'].astype(int)
    
    # Rolling metrics
    comm_bt['rolling_hit_ratio'] = comm_bt['correct'].rolling(window, min_periods=5).mean() * 100
    comm_bt['cumulative_hit_ratio'] = comm_bt['correct'].expanding().mean() * 100
    
    # Rolling Sharpe (using net PnL)
    comm_bt['rolling_sharpe'] = comm_bt['net_pnl'].rolling(window, min_periods=5).apply(
        lambda x: (x.mean() / (x.std() + 1e-8)) * np.sqrt(52) if len(x) > 0 and x.std() > 0 else 0
    )
    
    result = {
        'commodity': commodity,
        'window': window,
        'dates': [str(d.date()) for d in comm_bt['week']],
        'rolling_hit_ratio': [round(float(x), 2) if not pd.isna(x) else None for x in comm_bt['rolling_hit_ratio']],
        'cumulative_hit_ratio': [round(float(x), 2) for x in comm_bt['cumulative_hit_ratio']],
        'rolling_sharpe': [round(float(x), 2) if not pd.isna(x) else None for x in comm_bt['rolling_sharpe']],
        'overall_hit_ratio': round(float(comm_bt['correct'].mean() * 100), 2),
        'best_rolling_hit_ratio': round(float(comm_bt['rolling_hit_ratio'].max()), 2) if not comm_bt['rolling_hit_ratio'].isna().all() else 0,
        'worst_rolling_hit_ratio': round(float(comm_bt['rolling_hit_ratio'].min()), 2) if not comm_bt['rolling_hit_ratio'].isna().all() else 0,
        'total_predictions': len(comm_bt)
    }
    
    return jsonify(result)


@app.route('/api/analysis/multi_commodity_comparison', methods=['GET'])
def get_multi_commodity_comparison():
    """Get comparison metrics across all commodities."""
    backtest_df = load_data(BACKTEST_FILE, 'backtest')
    predictions_df = load_data(PREDICTIONS_FILE, 'predictions')
    
    if backtest_df is None:
        return jsonify({'error': 'Backtest not found'})
    
    commodities = backtest_df['commodity'].unique()
    comparison = []
    
    for commodity in commodities:
        # Backtest metrics
        bt = backtest_df[backtest_df['commodity'] == commodity]
        correct = ((bt['position'] > 0) & (bt['actual_return'] > 0)) | ((bt['position'] < 0) & (bt['actual_return'] < 0))
        hit_ratio = correct.mean() * 100
        
        # MAE from predictions if available
        mae = 0
        total_pred = len(bt)
        if predictions_df is not None:
            pred = predictions_df[predictions_df['commodity'] == commodity]
            if len(pred) > 0:
                mae = np.abs(pred['y_pred_ret'] - pred['y_true']).mean()
                total_pred = len(pred)
        
        # Sharpe and PnL
        returns = bt['net_pnl'].values
        sharpe = (np.mean(returns) / (np.std(returns) + 1e-8)) * np.sqrt(52) if len(returns) > 0 else 0
        final_pnl = bt['cumulative_pnl'].iloc[-1] if len(bt) > 0 else 0
        
        comparison.append({
            'commodity': commodity,
            'hit_ratio': round(float(hit_ratio), 2),
            'mae': round(float(mae), 4),
            'sharpe': round(float(sharpe), 2),
            'total_predictions': int(total_pred),
            'final_pnl': round(float(final_pnl), 4)
        })
    
    return jsonify({'comparison': comparison})


@app.route('/api/analysis/regime_performance', methods=['GET'])
def get_regime_performance():
    """Get performance by market regime (volatility)."""
    commodity = request.args.get('commodity', None)
    
    # Load backtest with volatility classification
    backtest_df = load_data(BACKTEST_FILE, 'backtest')
    if backtest_df is None:
        return jsonify({'error': 'Backtest data not found'})
    
    # Calculate rolling volatility
    backtest_df['week'] = pd.to_datetime(backtest_df['week'])
    backtest_df = backtest_df.sort_values(['commodity', 'week'])
    backtest_df['volatility'] = backtest_df.groupby('commodity')['actual_return'].transform(
        lambda x: x.rolling(20, min_periods=5).std()
    )
    
    # Classify regimes
    vol_median = backtest_df['volatility'].median()
    vol_75 = backtest_df['volatility'].quantile(0.75)
    backtest_df['regime'] = pd.cut(
        backtest_df['volatility'],
        bins=[0, vol_median, vol_75, np.inf],
        labels=['Low Volatility', 'Medium Volatility', 'High Volatility']
    )
    
    # Filter by commodity if specified
    if commodity:
        backtest_df = backtest_df[backtest_df['commodity'] == commodity]
    
    # Calculate regime metrics
    regime_stats = []
    for regime in ['Low Volatility', 'Medium Volatility', 'High Volatility']:
        regime_data = backtest_df[backtest_df['regime'] == regime]
        if len(regime_data) > 0:
            correct = (regime_data['position'] * regime_data['actual_return'] > 0).astype(int)
            
            # Calculate Sharpe, handling edge cases
            if len(regime_data) > 1 and regime_data['net_pnl'].std() > 0:
                sharpe = (regime_data['net_pnl'].mean() / regime_data['net_pnl'].std()) * np.sqrt(52)
                sharpe = round(float(sharpe), 2) if not np.isnan(sharpe) and not np.isinf(sharpe) else 0.0
            else:
                sharpe = 0.0
            
            regime_stats.append({
                'regime': regime,
                'count': int(len(regime_data)),
                'hit_ratio': round(float(correct.mean() * 100), 2),
                'avg_return': round(float(regime_data['net_pnl'].mean() * 100), 4),
                'sharpe': sharpe
            })
    
    return jsonify({
        'commodity': commodity if commodity else 'Portfolio',
        'regimes': regime_stats,
        'thresholds': {
            'low_medium': round(float(vol_median), 4),
            'medium_high': round(float(vol_75), 4)
        }
    })


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    print("="*60)
    print("TET-Weather Dashboard API")
    print("="*60)
    print(f"\nData directory: {DATA_DIR}")
    print(f"Predictions file: {PREDICTIONS_FILE} ({'✓' if PREDICTIONS_FILE.exists() else '✗'})")
    print(f"Signals file: {SIGNALS_FILE} ({'✓' if SIGNALS_FILE.exists() else '✗'})")
    print(f"Backtest file: {BACKTEST_FILE} ({'✓' if BACKTEST_FILE.exists() else '✗'})")
    print(f"Metrics file: {METRICS_FILE} ({'✓' if METRICS_FILE.exists() else '✗'})")
    print(f"\nStarting server on http://localhost:5000")
    print("="*60)
    
    app.run(debug=True, port=5000)
