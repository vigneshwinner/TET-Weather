"""
Forecast Dashboard API
Flask API serving predictions, SSI, and backtest data for the frontend.
"""

import os
from flask import Flask, jsonify
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import yaml

from src.api.routes import api_bp


def load_config():
    """Load configuration from config.yaml"""
    config_path = os.environ.get("CONFIG_PATH", "config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def create_app(config=None):
    """Application factory pattern"""
    app = Flask(__name__)
    
    # Load config
    if config is None:
        config = load_config()
    app.config["APP_CONFIG"] = config
    
    # Enable CORS for frontend
    CORS(app, origins=config.get("cors_origins", ["http://localhost:5173"]))
    
    # Rate limiting
    limiter = Limiter(
        key_func=get_remote_address,
        app=app,
        default_limits=["200 per day", "50 per hour"],
        storage_uri="memory://",
    )
    app.config["LIMITER"] = limiter
    
    # Register blueprints
    app.register_blueprint(api_bp)
    
    # Global error handlers
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({
            "error": "Not Found",
            "message": "The requested resource was not found",
            "status_code": 404
        }), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        return jsonify({
            "error": "Internal Server Error",
            "message": "An unexpected error occurred",
            "status_code": 500
        }), 500
    
    @app.errorhandler(429)
    def ratelimit_handler(error):
        return jsonify({
            "error": "Rate Limit Exceeded",
            "message": "Too many requests. Please try again later.",
            "status_code": 429
        }), 429
    
    return app


if __name__ == "__main__":
    app = create_app()
    debug = os.environ.get("FLASK_DEBUG", "true").lower() == "true"
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=debug, port=port, host="0.0.0.0")
