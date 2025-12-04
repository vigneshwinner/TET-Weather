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
DATA_DIR = BASE_DIR.parent  # Go up to TET-Weather root

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
    
    return jsonify({
        'commodity': commodity,
        'date': str(latest[week_col].date()),
        'predicted_return': round(float(latest['y_pred_ret']), 6),
        'direction_probability': round(float(prob), 4),
        'signal': signal,
        'confidence': round(abs(prob - 0.5) * 2, 4)
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
