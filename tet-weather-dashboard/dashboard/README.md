# TET-Weather Forecast Dashboard

A real-time dashboard for visualizing commodity price forecasts, trading signals, and backtest performance.

## Screenshots

The dashboard displays:
- **Current Signal**: Long/Short/Flat with direction probability
- **Performance Metrics**: Total return, Sharpe ratio, win rate, max drawdown
- **Equity Curve**: Cumulative PnL over time
- **Position History**: Historical positions by week
- **All Commodities Comparison**: Side-by-side equity curves
- **Model Evaluation**: MAE, R², accuracy, ROC-AUC per commodity

---

## Quick Start

### 1. Install Dependencies

**Backend (Flask API):**
```bash
cd dashboard/api
pip install -r requirements.txt
```

**Frontend (React):**
```bash
cd dashboard/frontend
npm install
```

### 2. Start the Servers

**Terminal 1 - Start API:**
```bash
cd dashboard/api
python app.py
```
API will run on http://localhost:5000

**Terminal 2 - Start Frontend:**
```bash
cd dashboard/frontend
npm run dev
```
Dashboard will open at http://localhost:5173

---

## Project Structure

```
dashboard/
├── api/
│   ├── app.py              # Flask API server
│   └── requirements.txt    # Python dependencies
│
└── frontend/
    ├── package.json        # Node dependencies
    ├── vite.config.js      # Vite configuration
    ├── tailwind.config.js  # Tailwind CSS config
    ├── index.html          # HTML entry point
    └── src/
        ├── main.jsx        # React entry point
        ├── index.css       # Global styles
        └── App.jsx         # Main application
```

---

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /api/health` | Health check |
| `GET /api/commodities` | List available commodities |
| `GET /api/forecast?commodity=X` | Get latest forecast |
| `GET /api/signals?commodity=X` | Get signal history |
| `GET /api/backtest?commodity=X` | Get backtest results |
| `GET /api/performance` | Get performance metrics |
| `GET /api/equity_curve` | Get equity curves |
| `GET /api/evaluation` | Get model evaluation |
| `POST /api/refresh` | Clear cache and reload |

---

## Data Files Required

The API reads from these files (relative to TET-Weather root):

```
TET-Weather/
├── reports/
│   ├── oos_predictions.csv        # Model predictions
│   ├── evaluation_results.csv     # Model evaluation
│   └── signals/
│       ├── weekly_signals.csv     # Trading signals
│       ├── backtest_results.csv   # Backtest PnL
│       └── performance_metrics.csv # Performance stats
```

Make sure you've run `run_eval.py` and `signal_generator.py` first!

---

## Configuration

### API Port
Edit `app.py`:
```python
app.run(debug=True, port=5000)
```

### Frontend Port
Edit `vite.config.js`:
```javascript
server: {
  port: 5173,
  ...
}
```

### Data Directory
If your data is elsewhere, edit the paths in `app.py`:
```python
DATA_DIR = Path('/your/custom/path')
```

---

## Troubleshooting

### "CORS Error"
Make sure both servers are running. The frontend proxies to the API.

### "Data not loading"
Check that your data files exist:
```bash
ls reports/oos_predictions.csv
ls reports/signals/backtest_results.csv
```

### "npm install fails"
Make sure Node.js is installed:
```bash
node --version  # Should be v18+
npm --version
```

### "Python dependencies missing"
```bash
pip install flask flask-cors pandas numpy
```

---

## Features

- **Dark Theme**: Modern dark UI with Tailwind CSS
- **Real-time Updates**: Refresh button to reload data
- **Interactive Charts**: Built with Recharts
- **Responsive Design**: Works on desktop and tablet
- **Commodity Selector**: Switch between 5 commodities
- **Performance Summary**: Key metrics at a glance

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Backend | Flask 3.0, Python |
| Frontend | React 18, Vite 5 |
| Styling | Tailwind CSS 3 |
| Charts | Recharts |
| Icons | Lucide React |

---

## License

MIT - Feel free to use and modify.
