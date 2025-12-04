"""
API Routes
Endpoints for forecasts, SSI, and backtest data.
"""

from datetime import datetime, timedelta
from flask import Blueprint, jsonify, request, current_app
from functools import wraps

from src.services.loader import DataLoader
from src.services.infer import ModelInference


api_bp = Blueprint("api", __name__)


def get_loader():
    """Get or create DataLoader instance"""
    if not hasattr(current_app, "_loader"):
        config = current_app.config["APP_CONFIG"]
        current_app._loader = DataLoader(config)
    return current_app._loader


def get_inference():
    """Get or create ModelInference instance"""
    if not hasattr(current_app, "_inference"):
        config = current_app.config["APP_CONFIG"]
        current_app._inference = ModelInference(config)
    return current_app._inference


def validate_params(required_params):
    """Decorator for parameter validation"""
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            missing = []
            for param in required_params:
                if not request.args.get(param):
                    missing.append(param)
            if missing:
                return jsonify({
                    "error": "Validation Error",
                    "message": f"Missing required parameters: {', '.join(missing)}",
                    "status_code": 400
                }), 400
            return f(*args, **kwargs)
        return wrapper
    return decorator


def validate_date(date_str, param_name):
    """Validate date format YYYY-MM-DD"""
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        return None


def validate_commodity(commodity):
    """Validate commodity is supported"""
    loader = get_loader()
    supported = loader.get_commodities()
    return commodity.upper() in [c.upper() for c in supported]


# =============================================================================
# Health Check
# =============================================================================

@api_bp.route("/health", methods=["GET"])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0"
    })


# =============================================================================
# Commodities
# =============================================================================

@api_bp.route("/commodities", methods=["GET"])
def commodities():
    """Get list of supported commodities/tickers"""
    try:
        loader = get_loader()
        commodity_list = loader.get_commodities()
        return jsonify({
            "commodities": commodity_list,
            "count": len(commodity_list)
        })
    except Exception as e:
        return jsonify({
            "error": "Server Error",
            "message": str(e),
            "status_code": 500
        }), 500


# =============================================================================
# Forecast
# =============================================================================

@api_bp.route("/forecast", methods=["GET"])
@validate_params(["commodity"])
def forecast():
    """
    Get forecast for a commodity.
    
    Query params:
        commodity: Ticker symbol (required)
        week: Target week in YYYY-MM-DD format (optional, defaults to next week)
    
    Returns:
        predicted_return: Expected return magnitude
        direction_probability: Probability of positive direction (0-1)
        confidence: Model confidence level
        model_version: Version of the model used
    """
    commodity = request.args.get("commodity").upper()
    week = request.args.get("week")
    
    # Validate commodity
    if not validate_commodity(commodity):
        return jsonify({
            "error": "Validation Error",
            "message": f"Unsupported commodity: {commodity}",
            "status_code": 400
        }), 400
    
    # Validate/default week
    if week:
        week_dt = validate_date(week, "week")
        if not week_dt:
            return jsonify({
                "error": "Validation Error",
                "message": "Invalid date format for 'week'. Use YYYY-MM-DD",
                "status_code": 400
            }), 400
    else:
        # Default to next week (next Monday)
        today = datetime.now()
        days_ahead = 7 - today.weekday()
        week_dt = today + timedelta(days=days_ahead)
    
    try:
        inference = get_inference()
        result = inference.predict(commodity, week_dt)
        
        return jsonify({
            "commodity": commodity,
            "week": week_dt.strftime("%Y-%m-%d"),
            "predicted_return": result["predicted_return"],
            "direction_probability": result["direction_probability"],
            "confidence": result["confidence"],
            "model_version": result.get("model_version", "unknown"),
            "generated_at": datetime.utcnow().isoformat()
        })
    except Exception as e:
        return jsonify({
            "error": "Prediction Error",
            "message": str(e),
            "status_code": 500
        }), 500


# =============================================================================
# SSI (Signal Strength Index)
# =============================================================================

@api_bp.route("/ssi", methods=["GET"])
@validate_params(["commodity"])
def ssi():
    """
    Get SSI time series for a commodity.
    
    Query params:
        commodity: Ticker symbol (required)
        start: Start date YYYY-MM-DD (optional, defaults to 1 year ago)
        end: End date YYYY-MM-DD (optional, defaults to today)
    
    Returns:
        data: Array of {date, ssi_value, components}
    """
    commodity = request.args.get("commodity").upper()
    start = request.args.get("start")
    end = request.args.get("end")
    
    # Validate commodity
    if not validate_commodity(commodity):
        return jsonify({
            "error": "Validation Error",
            "message": f"Unsupported commodity: {commodity}",
            "status_code": 400
        }), 400
    
    # Validate/default dates
    if end:
        end_dt = validate_date(end, "end")
        if not end_dt:
            return jsonify({
                "error": "Validation Error",
                "message": "Invalid date format for 'end'. Use YYYY-MM-DD",
                "status_code": 400
            }), 400
    else:
        end_dt = datetime.now()
    
    if start:
        start_dt = validate_date(start, "start")
        if not start_dt:
            return jsonify({
                "error": "Validation Error",
                "message": "Invalid date format for 'start'. Use YYYY-MM-DD",
                "status_code": 400
            }), 400
    else:
        start_dt = end_dt - timedelta(days=365)
    
    # Validate date range
    if start_dt >= end_dt:
        return jsonify({
            "error": "Validation Error",
            "message": "Start date must be before end date",
            "status_code": 400
        }), 400
    
    try:
        loader = get_loader()
        ssi_data = loader.get_ssi(commodity, start_dt, end_dt)
        
        return jsonify({
            "commodity": commodity,
            "start": start_dt.strftime("%Y-%m-%d"),
            "end": end_dt.strftime("%Y-%m-%d"),
            "data": ssi_data,
            "count": len(ssi_data)
        })
    except Exception as e:
        return jsonify({
            "error": "Data Error",
            "message": str(e),
            "status_code": 500
        }), 500


# =============================================================================
# Backtest Summary
# =============================================================================

@api_bp.route("/backtest/summary", methods=["GET"])
@validate_params(["commodity"])
def backtest_summary():
    """
    Get backtest summary metrics for a commodity.
    
    Query params:
        commodity: Ticker symbol (required)
    
    Returns:
        metrics: Dict of performance metrics (sharpe, returns, etc.)
        equity_curve: Recent equity curve points
    """
    commodity = request.args.get("commodity").upper()
    
    # Validate commodity
    if not validate_commodity(commodity):
        return jsonify({
            "error": "Validation Error",
            "message": f"Unsupported commodity: {commodity}",
            "status_code": 400
        }), 400
    
    try:
        loader = get_loader()
        summary = loader.get_backtest_summary(commodity)
        
        return jsonify({
            "commodity": commodity,
            "metrics": summary["metrics"],
            "equity_curve": summary["equity_curve"],
            "last_updated": summary.get("last_updated", datetime.utcnow().isoformat())
        })
    except Exception as e:
        return jsonify({
            "error": "Data Error",
            "message": str(e),
            "status_code": 500
        }), 500


# =============================================================================
# Scenario Simulation (Optional)
# =============================================================================

@api_bp.route("/simulate", methods=["POST"])
def simulate_scenario():
    """
    Run a simulated scenario (optional endpoint).
    
    Body:
        commodity: Ticker symbol
        scenario: Scenario parameters
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                "error": "Validation Error",
                "message": "Request body is required",
                "status_code": 400
            }), 400
        
        commodity = data.get("commodity", "").upper()
        scenario = data.get("scenario", {})
        
        if not commodity:
            return jsonify({
                "error": "Validation Error",
                "message": "Commodity is required",
                "status_code": 400
            }), 400
        
        if not validate_commodity(commodity):
            return jsonify({
                "error": "Validation Error",
                "message": f"Unsupported commodity: {commodity}",
                "status_code": 400
            }), 400
        
        inference = get_inference()
        result = inference.simulate(commodity, scenario)
        
        return jsonify({
            "commodity": commodity,
            "scenario": scenario,
            "result": result,
            "simulated_at": datetime.utcnow().isoformat()
        })
    except Exception as e:
        return jsonify({
            "error": "Simulation Error",
            "message": str(e),
            "status_code": 500
        }), 500


# =============================================================================
# API Documentation
# =============================================================================

@api_bp.route("/docs", methods=["GET"])
def docs():
    """Return API documentation"""
    return jsonify({
        "name": "Forecast Dashboard API",
        "version": "1.0.0",
        "endpoints": [
            {
                "path": "/health",
                "method": "GET",
                "description": "Health check",
                "params": []
            },
            {
                "path": "/commodities",
                "method": "GET",
                "description": "List supported commodities",
                "params": []
            },
            {
                "path": "/forecast",
                "method": "GET",
                "description": "Get forecast for a commodity",
                "params": [
                    {"name": "commodity", "required": True, "description": "Ticker symbol"},
                    {"name": "week", "required": False, "description": "Target week (YYYY-MM-DD)"}
                ]
            },
            {
                "path": "/ssi",
                "method": "GET",
                "description": "Get SSI time series",
                "params": [
                    {"name": "commodity", "required": True, "description": "Ticker symbol"},
                    {"name": "start", "required": False, "description": "Start date (YYYY-MM-DD)"},
                    {"name": "end", "required": False, "description": "End date (YYYY-MM-DD)"}
                ]
            },
            {
                "path": "/backtest/summary",
                "method": "GET",
                "description": "Get backtest metrics and equity curve",
                "params": [
                    {"name": "commodity", "required": True, "description": "Ticker symbol"}
                ]
            },
            {
                "path": "/simulate",
                "method": "POST",
                "description": "Run scenario simulation",
                "body": {
                    "commodity": "Ticker symbol",
                    "scenario": "Scenario parameters object"
                }
            }
        ]
    })
