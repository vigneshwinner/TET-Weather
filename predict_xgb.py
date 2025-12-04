"""
XGBoost Prediction Inference Pipeline
Load trained models and make predictions on new data
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class XGBoostPredictor:
    """
    Loads trained XGBoost models and makes predictions.
    """
    
    def __init__(self, commodity: str, artifacts_dir: str = 'cleaned_data/xgboost_artifacts'):
        """
        Initialize predictor for a specific commodity.
        
        Args:
            commodity: One of ['Brent', 'Henry_Hub', 'Power', 'Copper', 'Corn']
            artifacts_dir: Directory containing trained model artifacts
        """
        self.commodity = commodity
        self.artifacts_dir = Path(artifacts_dir)
        self.models = self._load_models()
        
        if not self.models:
            raise ValueError(f"No models found for {commodity} in {artifacts_dir}")
        
        print(f"✓ Loaded {len(self.models)} models for {commodity}")
    
    def _load_models(self) -> List[Dict]:
        """Load all model artifacts for this commodity."""
        models = []
        pattern = f"{self.commodity}_fold*_xgb.joblib"
        
        for filepath in sorted(self.artifacts_dir.glob(pattern)):
            artifacts = joblib.load(filepath)
            models.append(artifacts)
        
        return models
    
    def prepare_features(self, features_df: pd.DataFrame) -> np.ndarray:
        """
        Prepare features using the same preprocessing as training.
        
        Args:
            features_df: DataFrame with feature columns matching training data
        
        Returns:
            Scaled feature array
        """
        if not self.models:
            raise ValueError("No models loaded")
        
        # Use the most recent model's feature names and scaler
        latest_model = self.models[-1]
        feature_names = latest_model['feature_names']
        scaler = latest_model['scaler']
        
        # Ensure all required features are present
        missing_features = set(feature_names) - set(features_df.columns)
        if missing_features:
            print(f"⚠️  Warning: Missing features will be filled with 0: {missing_features}")
            for feat in missing_features:
                features_df[feat] = 0
        
        # Select and order features to match training
        X = features_df[feature_names].values
        
        # Scale features
        X_scaled = scaler.transform(X)
        
        return X_scaled
    
    def predict_return(self, features_df: pd.DataFrame, use_ensemble: bool = True) -> np.ndarray:
        """
        Predict next-week log returns.
        
        Args:
            features_df: DataFrame with feature columns
            use_ensemble: If True, average predictions from all models. If False, use latest model.
        
        Returns:
            Array of predicted returns
        """
        X_scaled = self.prepare_features(features_df)
        
        if use_ensemble:
            predictions = []
            for model_artifacts in self.models:
                reg_model = model_artifacts['reg_model']
                pred = reg_model.predict(X_scaled)
                predictions.append(pred)
            
            # Average predictions
            return np.mean(predictions, axis=0)
        else:
            # Use latest model only
            reg_model = self.models[-1]['reg_model']
            return reg_model.predict(X_scaled)
    
    def predict_direction(self, features_df: pd.DataFrame, use_ensemble: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict next-week direction (up/down).
        
        Args:
            features_df: DataFrame with feature columns
            use_ensemble: If True, average probabilities from all models. If False, use latest model.
        
        Returns:
            Tuple of (predicted_labels, predicted_probabilities)
        """
        X_scaled = self.prepare_features(features_df)
        
        if use_ensemble:
            probabilities = []
            for model_artifacts in self.models:
                clf_model = model_artifacts['clf_model']
                proba = clf_model.predict_proba(X_scaled)[:, 1]
                probabilities.append(proba)
            
            # Average probabilities
            avg_proba = np.mean(probabilities, axis=0)
            labels = (avg_proba > 0.5).astype(int)
            
            return labels, avg_proba
        else:
            # Use latest model only
            clf_model = self.models[-1]['clf_model']
            proba = clf_model.predict_proba(X_scaled)[:, 1]
            labels = (proba > 0.5).astype(int)
            
            return labels, proba
    
    def predict_both(self, features_df: pd.DataFrame, use_ensemble: bool = True) -> pd.DataFrame:
        """
        Predict both return magnitude and direction.
        
        Args:
            features_df: DataFrame with feature columns
            use_ensemble: If True, average predictions from all models
        
        Returns:
            DataFrame with predictions
        """
        returns = self.predict_return(features_df, use_ensemble=use_ensemble)
        directions, probas = self.predict_direction(features_df, use_ensemble=use_ensemble)
        
        results = pd.DataFrame({
            'predicted_return': returns,
            'predicted_direction': directions,
            'direction_probability': probas,
            'direction_from_return': (returns > 0).astype(int)
        })
        
        return results
    
    def get_feature_importance(self, importance_type: str = 'gain', top_n: int = 20) -> pd.DataFrame:
        """
        Get feature importance from the most recent model.
        
        Args:
            importance_type: 'gain', 'weight', or 'cover'
            top_n: Number of top features to return
        
        Returns:
            DataFrame with feature names and importance scores
        """
        latest_model = self.models[-1]
        reg_model = latest_model['reg_model']
        feature_names = latest_model['feature_names']
        
        importance = reg_model.get_booster().get_score(importance_type=importance_type)
        
        importance_df = pd.DataFrame([
            {'feature': feature_names[int(k.replace('f', ''))], importance_type: v}
            for k, v in importance.items()
        ]).sort_values(importance_type, ascending=False).head(top_n)
        
        return importance_df
    
    def get_model_info(self) -> Dict:
        """Get information about loaded models."""
        latest_model = self.models[-1]
        
        return {
            'commodity': self.commodity,
            'n_models': len(self.models),
            'n_features': len(latest_model['feature_names']),
            'latest_fold': latest_model['fold'],
            'latest_test_week': latest_model['test_week'],
            'scale_pos_weight': latest_model.get('scale_pos_weight', 1.0),
            'reg_params': latest_model['best_params_reg'],
            'clf_params': latest_model['best_params_clf']
        }


# ============================================================================
# Demo Usage
# ============================================================================

if __name__ == '__main__':
    print("="*80)
    print("XGBOOST PREDICTION DEMO")
    print("="*80)
    
    # Load feature data for demonstration
    print("\n📂 Loading feature data...")
    
    interactions_df = pd.read_csv('cleaned_data/weather_eia_interactions.csv')
    interactions_df['date'] = pd.to_datetime(interactions_df['date'])
    interactions_df['week'] = interactions_df['date'] - pd.to_timedelta(interactions_df['date'].dt.dayofweek.astype('int64'), unit='D')  # type: ignore[attr-defined]
    
    degree_days_df = pd.read_csv('cleaned_data/degree_days.csv')
    degree_days_df['week'] = pd.to_datetime(degree_days_df['week'])
    
    commodity_map = {'Crude_Oil': 'Brent', 'Natural_Gas': 'Henry_Hub', 'Power': 'Power', 'Copper': 'Copper', 'Corn': 'Corn'}
    interactions_df['commodity_mapped'] = interactions_df['commodity'].map(commodity_map)
    
    feature_cols = [col for col in interactions_df.columns if 
                    col not in ['date', 'week', 'region', 'commodity', 'commodity_mapped', 'date_eia']]
    
    interactions_weekly = interactions_df.groupby(['week', 'commodity_mapped'])[feature_cols].mean().reset_index()
    interactions_weekly = interactions_weekly.rename(columns={'commodity_mapped': 'commodity'})
    interactions_weekly = interactions_weekly[interactions_weekly['commodity'].notna()]
    
    dd_weekly = degree_days_df.groupby('week').agg({
        'hdd_weekly_sum': 'mean', 'cdd_weekly_sum': 'mean',
        'hdd_7day_avg': 'mean', 'cdd_7day_avg': 'mean',
        'hdd_14day_avg': 'mean', 'cdd_14day_avg': 'mean',
        'hdd_30day_avg': 'mean', 'cdd_30day_avg': 'mean'
    }).reset_index()
    
    # Get most recent week for each commodity
    for commodity in ['Brent', 'Henry_Hub', 'Power', 'Copper', 'Corn']:
        print(f"\n{'='*80}")
        print(f"COMMODITY: {commodity}")
        print(f"{'='*80}")
        
        try:
            predictor = XGBoostPredictor(commodity)
            
            # Get model info
            info = predictor.get_model_info()
            print(f"\n📊 Model Info:")
            print(f"  Number of models: {info['n_models']}")
            print(f"  Number of features: {info['n_features']}")
            print(f"  Latest fold: {info['latest_fold']}")
            print(f"  Latest test week: {info['latest_test_week']}")
            print(f"  Scale pos weight: {info['scale_pos_weight']:.2f}")
            
            # Prepare features for most recent week
            if commodity in ['Copper', 'Corn']:
                # Copper and Corn only have degree days
                recent_week = dd_weekly['week'].max()
                features_df = dd_weekly[dd_weekly['week'] == recent_week].copy()
                
                # Add missing features as zeros
                for col in feature_cols:
                    if col not in features_df.columns:
                        features_df[col] = 0
            else:
                # Brent and Henry_Hub have interactions
                commodity_data = interactions_weekly[interactions_weekly['commodity'] == commodity]
                recent_week = commodity_data['week'].max()
                features_df = commodity_data[commodity_data['week'] == recent_week].copy()
                
                # Merge degree days
                features_df = features_df.merge(dd_weekly, on='week', how='left')
            
            features_df = features_df.fillna(0)
            
            print(f"\n🔮 Making predictions for week: {recent_week.date()}")
            
            # Make predictions
            predictions = predictor.predict_both(features_df, use_ensemble=True)
            
            print(f"\n📈 Predictions (ensemble of {info['n_models']} models):")
            print(f"  Predicted return: {predictions['predicted_return'].values[0]:.4f}")
            print(f"  Direction: {'UP ⬆️' if predictions['predicted_direction'].values[0] == 1 else 'DOWN ⬇️'}")
            print(f"  Direction probability: {predictions['direction_probability'].values[0]:.2%}")
            print(f"  Direction from return: {'UP ⬆️' if predictions['direction_from_return'].values[0] == 1 else 'DOWN ⬇️'}")
            
            # Get feature importance
            print(f"\n🔍 Top 10 Features by Gain:")
            importance = predictor.get_feature_importance(importance_type='gain', top_n=10)
            for idx, row in importance.iterrows():
                print(f"  {row['feature']}: {row['gain']:.2f}")
            
        except Exception as e:
            print(f"⚠️  Error: {e}")
    
    print(f"\n" + "="*80)
    print("✅ DEMO COMPLETE")
    print("="*80 + "\n")
