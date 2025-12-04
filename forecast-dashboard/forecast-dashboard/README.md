# Forecast Dashboard

A full-stack application for displaying energy/commodity forecasts, Signal Strength Index (SSI), and backtest performance.

## Architecture

```
forecast-dashboard/
├── api/                    # Flask backend
│   ├── app.py             # Main entry point
│   ├── config.yaml        # Configuration
│   ├── src/
│   │   ├── api/
│   │   │   └── routes.py  # API endpoints
│   │   └── services/
│   │       ├── loader.py  # Data loading
│   │       └── infer.py   # Model inference
│   ├── requirements.txt
│   └── openapi.yaml       # API specification
│
└── frontend/              # React + Vite frontend
    ├── src/
    │   ├── components/    # React components
    │   ├── services/      # API client
    │   └── App.jsx        # Main app
    ├── package.json
    └── vite.config.js
```

## Quick Start

### 1. Backend Setup

```bash
cd api

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create data directories (optional - will use mock data if not present)
mkdir -p data/ssi data/backtest data/features models/registry

# Run the server
python app.py
```

The API will be available at `http://localhost:5000`

### 2. Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Copy environment file
cp .env.example .env

# Run development server
npm run dev
```

The frontend will be available at `http://localhost:5173`

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/commodities` | GET | List supported commodities |
| `/forecast` | GET | Get forecast for a commodity |
| `/ssi` | GET | Get SSI time series |
| `/backtest/summary` | GET | Get backtest metrics |
| `/simulate` | POST | Run scenario simulation |
| `/docs` | GET | API documentation |

### Example Requests

```bash
# Health check
curl http://localhost:5000/health

# Get commodities
curl http://localhost:5000/commodities

# Get forecast
curl "http://localhost:5000/forecast?commodity=NG"

# Get SSI data
curl "http://localhost:5000/ssi?commodity=NG&start=2024-01-01&end=2024-12-01"

# Get backtest summary
curl "http://localhost:5000/backtest/summary?commodity=NG"
```

## Configuration

### Backend (config.yaml)

```yaml
data_dir: "data"
model_registry_path: "models/registry"
commodities:
  - "NG"
  - "CL"
  - "ERCOT"
cors_origins:
  - "http://localhost:5173"
cache_ttl_seconds: 300
```

### Frontend (.env)

```
VITE_API_BASE_URL=http://localhost:5000
VITE_ENABLE_SIMULATION=true
```

## Data Files

The API expects data in the following locations:

```
data/
├── ssi/
│   ├── ng_ssi.csv          # SSI time series (date, ssi_value, component_*)
│   └── cl_ssi.csv
├── backtest/
│   ├── ng_summary.json     # Backtest summary
│   ├── ng_equity.csv       # Equity curve (date, value)
│   └── cl_summary.json
└── features/
    └── ng_features.csv     # Features for inference

models/
└── registry/
    └── ng/
        ├── champion.json   # Champion model marker
        └── xgb_model.pkl   # Trained model
```

If data files are not present, the API will return mock data for development.

## Integrating Your Models

1. **Save your trained XGBoost model**:

```python
import pickle

model_bundle = {
    "magnitude_model": magnitude_booster,
    "direction_model": direction_booster,
    "metadata": {
        "version": "v1.0",
        "trained_at": "2024-12-01",
        "features": feature_names,
    }
}

with open("models/registry/ng/xgb_model.pkl", "wb") as f:
    pickle.dump(model_bundle, f)
```

2. **Create champion marker**:

```json
{
  "model_name": "xgb_model.pkl",
  "promoted_at": "2024-12-01T00:00:00Z",
  "metrics": {
    "sharpe": 1.5,
    "accuracy": 0.58
  }
}
```

## Production Deployment

### Backend

```bash
# Using gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:create_app()
```

### Frontend

```bash
# Build for production
npm run build

# Preview build
npm run preview

# Serve with nginx, etc.
```

## Development

### Adding New Commodities

1. Add ticker to `config.yaml` commodities list
2. Add display name to `Header.jsx` `getCommodityName()` function
3. Ensure data files exist or mock data will be used

### Customizing Charts

Charts use Plotly.js. Modify layout/config in:
- `SSIPanel.jsx` - SSI time series
- `PerformancePanel.jsx` - Equity curve

## License

MIT
