"""
Model Inference Service
Handles loading models from registry and making predictions.
"""

import os
import json
import pickle
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np

logger = logging.getLogger(__name__)


class ModelInference:
    """
    Handles model loading and inference for forecasts.
    Integrates with the model registry from Phase 3.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_registry_path = Path(config.get("model_registry_path", "models/registry"))
        self.models = {}
        self._model_metadata = {}
        
        # Import data loader for features
        from src.services.loader import DataLoader
        self.loader = DataLoader(config)
    
    def _load_model(self, commodity: str) -> Optional[Any]:
        """
        Load the champion model for a commodity from the registry.
        """
        if commodity in self.models:
            return self.models[commodity]
        
        # Look for champion model in registry
        model_dir = self.model_registry_path / commodity.lower()
        
        # Try to find champion marker or latest model
        champion_file = model_dir / "champion.json"
        
        if champion_file.exists():
            with open(champion_file, "r") as f:
                champion_info = json.load(f)
            model_name = champion_info.get("model_name", "xgb_model.pkl")
            model_path = model_dir / model_name
        else:
            # Fall back to default model name
            model_path = model_dir / "xgb_model.pkl"
        
        if not model_path.exists():
            # Try alternative paths
            alt_paths = [
                model_dir / "model.pkl",
                model_dir / "magnitude_model.pkl",
                self.model_registry_path / f"{commodity.lower()}_xgb.pkl",
            ]
            
            for alt in alt_paths:
                if alt.exists():
                    model_path = alt
                    break
            else:
                logger.warning(f"No model found for {commodity}")
                return None
        
        try:
            with open(model_path, "rb") as f:
                model_data = pickle.load(f)
            
            # Handle different model formats
            if isinstance(model_data, dict):
                # Model bundle with magnitude and direction models
                self.models[commodity] = model_data
                self._model_metadata[commodity] = model_data.get("metadata", {})
            else:
                # Single model object
                self.models[commodity] = {"model": model_data}
                self._model_metadata[commodity] = {}
            
            logger.info(f"Loaded model for {commodity} from {model_path}")
            return self.models[commodity]
            
        except Exception as e:
            logger.error(f"Failed to load model for {commodity}: {e}")
            return None
    
    def predict(self, commodity: str, target_date: datetime) -> Dict[str, Any]:
        """
        Generate forecast for a commodity and target date.
        
        Returns:
            predicted_return: Expected return magnitude
            direction_probability: Probability of positive direction
            confidence: Model confidence level
            model_version: Version string
        """
        model_data = self._load_model(commodity)
        
        if model_data is None:
            # Return mock prediction for development
            logger.warning(f"Using mock prediction for {commodity}")
            return self._mock_prediction(commodity, target_date)
        
        # Load features for the prediction date
        features_df = self.loader.load_features(commodity, target_date)
        
        if features_df is None:
            logger.warning(f"No features available for {commodity} at {target_date}")
            return self._mock_prediction(commodity, target_date)
        
        try:
            # Extract feature matrix
            feature_cols = [c for c in features_df.columns if c not in ["date", "target", "direction"]]
            X = features_df[feature_cols].values
            
            # Get magnitude prediction
            if "magnitude_model" in model_data:
                magnitude_model = model_data["magnitude_model"]
                predicted_return = float(magnitude_model.predict(X)[0])
            elif "model" in model_data:
                model = model_data["model"]
                predicted_return = float(model.predict(X)[0])
            else:
                predicted_return = 0.0
            
            # Get direction prediction
            if "direction_model" in model_data:
                direction_model = model_data["direction_model"]
                direction_proba = float(direction_model.predict_proba(X)[0, 1])
            else:
                # Infer direction from magnitude
                direction_proba = 0.5 + 0.5 * np.tanh(predicted_return * 10)
            
            # Calculate confidence based on probability distance from 0.5
            confidence = abs(direction_proba - 0.5) * 2
            
            # Get model version
            metadata = self._model_metadata.get(commodity, {})
            model_version = metadata.get("version", metadata.get("trained_at", "unknown"))
            
            return {
                "predicted_return": round(predicted_return, 4),
                "direction_probability": round(direction_proba, 4),
                "confidence": round(confidence, 4),
                "model_version": str(model_version)
            }
            
        except Exception as e:
            logger.error(f"Prediction failed for {commodity}: {e}")
            return self._mock_prediction(commodity, target_date)
    
    def _mock_prediction(self, commodity: str, target_date: datetime) -> Dict[str, Any]:
        """Generate mock prediction for development/testing"""
        np.random.seed(hash(f"{commodity}{target_date.isoformat()}") % 2**32)
        
        predicted_return = np.random.randn() * 0.05
        direction_proba = 0.5 + np.random.randn() * 0.2
        direction_proba = np.clip(direction_proba, 0.1, 0.9)
        
        return {
            "predicted_return": round(float(predicted_return), 4),
            "direction_probability": round(float(direction_proba), 4),
            "confidence": round(float(abs(direction_proba - 0.5) * 2), 4),
            "model_version": "mock-v1.0"
        }
    
    def simulate(self, commodity: str, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a simulated scenario with modified inputs.
        
        Args:
            commodity: Ticker symbol
            scenario: Dict of feature overrides or scenario parameters
        
        Returns:
            Simulation results including baseline vs scenario comparison
        """
        model_data = self._load_model(commodity)
        
        if model_data is None:
            return {
                "status": "error",
                "message": f"No model available for {commodity}"
            }
        
        try:
            # Get baseline prediction
            baseline = self.predict(commodity, datetime.now())
            
            # For now, return mock scenario results
            # In production, this would modify features based on scenario params
            scenario_factor = scenario.get("shock_magnitude", 1.0)
            
            return {
                "baseline": baseline,
                "scenario": {
                    "predicted_return": baseline["predicted_return"] * scenario_factor,
                    "direction_probability": baseline["direction_probability"],
                    "confidence": baseline["confidence"] * 0.8,  # Reduced confidence for scenarios
                },
                "delta": {
                    "return_change": baseline["predicted_return"] * (scenario_factor - 1),
                    "confidence_change": -baseline["confidence"] * 0.2,
                },
                "scenario_params": scenario
            }
            
        except Exception as e:
            logger.error(f"Simulation failed for {commodity}: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    def get_model_info(self, commodity: str) -> Dict[str, Any]:
        """Get metadata about the loaded model"""
        if commodity not in self._model_metadata:
            self._load_model(commodity)
        
        return self._model_metadata.get(commodity, {})
