"""
Model Registry
Lightweight model registry for tracking trained models and selecting champions.

Issue #17: Unified Evaluation and Model Registry
"""

import json
import yaml
import hashlib
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict, field
import shutil


@dataclass
class ModelSpec:
    """Specification for a registered model."""
    model_name: str
    version: str
    commodity: str
    target: str  # 'magnitude' or 'direction' or 'both'
    
    # Training info
    training_data_start: str
    training_data_end: str
    n_training_samples: int
    feature_list_hash: str
    
    # Metrics
    metrics: Dict[str, float]
    
    # Artifacts
    artifact_paths: Dict[str, str]
    
    # Metadata
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    git_commit: Optional[str] = None
    notes: Optional[str] = None
    is_champion: bool = False
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ModelSpec':
        return cls(**data)
    
    def to_yaml(self) -> str:
        return yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=False)
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


class ModelRegistry:
    """
    Lightweight model registry for tracking trained models.
    
    Structure:
        registry_path/
        ├── registry.yaml          # Main registry index
        ├── models/
        │   ├── Henry_Hub/
        │   │   ├── xgb_v1.yaml
        │   │   ├── xgb_v2.yaml
        │   │   └── champion.yaml  # Symlink/copy of current champion
        │   └── Brent/
        │       └── ...
        └── artifacts/
            └── ...
    """
    
    def __init__(self, registry_path: str = 'models/registry'):
        self.registry_path = Path(registry_path)
        self.models_path = self.registry_path / 'models'
        self.artifacts_path = self.registry_path / 'artifacts'
        
        # Create directories
        self.registry_path.mkdir(parents=True, exist_ok=True)
        self.models_path.mkdir(exist_ok=True)
        self.artifacts_path.mkdir(exist_ok=True)
        
        # Load or create registry index
        self.index_path = self.registry_path / 'registry.yaml'
        self.index = self._load_index()
    
    def _load_index(self) -> Dict:
        """Load registry index from disk."""
        if self.index_path.exists():
            with open(self.index_path, 'r') as f:
                return yaml.safe_load(f) or {}
        return {
            'created_at': datetime.utcnow().isoformat(),
            'models': {},
            'champions': {}
        }
    
    def _save_index(self):
        """Save registry index to disk."""
        self.index['updated_at'] = datetime.utcnow().isoformat()
        with open(self.index_path, 'w') as f:
            yaml.dump(self.index, f, default_flow_style=False)
    
    def _get_git_commit(self) -> Optional[str]:
        """Get current git commit hash."""
        try:
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                capture_output=True,
                text=True,
                cwd=self.registry_path.parent
            )
            if result.returncode == 0:
                return result.stdout.strip()[:8]
        except Exception:
            pass
        return None
    
    def _hash_feature_list(self, features: List[str]) -> str:
        """Create hash of feature list for tracking."""
        feature_str = ','.join(sorted(features))
        return hashlib.md5(feature_str.encode()).hexdigest()[:12]
    
    def register_model(
        self,
        model_name: str,
        version: str,
        commodity: str,
        target: str,
        training_data_start: str,
        training_data_end: str,
        n_training_samples: int,
        feature_list: List[str],
        metrics: Dict[str, float],
        artifact_paths: Dict[str, str],
        notes: Optional[str] = None
    ) -> ModelSpec:
        """
        Register a new model in the registry.
        
        Args:
            model_name: Name of the model (e.g., 'xgb', 'ridge')
            version: Version string (e.g., 'v1', 'v1.1')
            commodity: Commodity name
            target: Target type ('magnitude', 'direction', or 'both')
            training_data_start: Start date of training data
            training_data_end: End date of training data
            n_training_samples: Number of training samples
            feature_list: List of feature names
            metrics: Dictionary of evaluation metrics
            artifact_paths: Dictionary of artifact paths
            notes: Optional notes
        
        Returns:
            ModelSpec for the registered model
        """
        # Create model spec
        spec = ModelSpec(
            model_name=model_name,
            version=version,
            commodity=commodity,
            target=target,
            training_data_start=training_data_start,
            training_data_end=training_data_end,
            n_training_samples=n_training_samples,
            feature_list_hash=self._hash_feature_list(feature_list),
            metrics=metrics,
            artifact_paths=artifact_paths,
            git_commit=self._get_git_commit(),
            notes=notes
        )
        
        # Create commodity directory
        commodity_path = self.models_path / commodity
        commodity_path.mkdir(exist_ok=True)
        
        # Save spec file
        spec_filename = f'{model_name}_{version}.yaml'
        spec_path = commodity_path / spec_filename
        
        with open(spec_path, 'w') as f:
            f.write(spec.to_yaml())
        
        # Update index
        if commodity not in self.index['models']:
            self.index['models'][commodity] = {}
        
        model_key = f'{model_name}_{version}'
        self.index['models'][commodity][model_key] = {
            'spec_path': str(spec_path),
            'registered_at': spec.created_at,
            'metrics_summary': {
                k: v for k, v in metrics.items() 
                if k in ['mae', 'rmse', 'r2', 'accuracy', 'roc_auc', 'brier_score']
            }
        }
        
        self._save_index()
        
        print(f"✓ Registered model: {commodity}/{model_name}_{version}")
        
        return spec
    
    def get_model(self, commodity: str, model_name: str, version: str) -> Optional[ModelSpec]:
        """Get a specific model spec."""
        model_key = f'{model_name}_{version}'
        
        if commodity not in self.index['models']:
            return None
        if model_key not in self.index['models'][commodity]:
            return None
        
        spec_path = Path(self.index['models'][commodity][model_key]['spec_path'])
        
        if spec_path.exists():
            with open(spec_path, 'r') as f:
                data = yaml.safe_load(f)
            return ModelSpec.from_dict(data)
        
        return None
    
    def list_models(self, commodity: Optional[str] = None) -> List[Dict]:
        """List all registered models."""
        models = []
        
        commodities = [commodity] if commodity else self.index['models'].keys()
        
        for comm in commodities:
            if comm not in self.index['models']:
                continue
            
            for model_key, model_info in self.index['models'][comm].items():
                models.append({
                    'commodity': comm,
                    'model_key': model_key,
                    'registered_at': model_info.get('registered_at'),
                    'metrics': model_info.get('metrics_summary', {}),
                    'is_champion': self.index['champions'].get(comm) == model_key
                })
        
        return models
    
    def set_champion(self, commodity: str, model_name: str, version: str) -> bool:
        """
        Set a model as the champion for a commodity.
        
        Args:
            commodity: Commodity name
            model_name: Model name
            version: Model version
        
        Returns:
            True if successful
        """
        model_key = f'{model_name}_{version}'
        
        if commodity not in self.index['models']:
            print(f"❌ Commodity '{commodity}' not found in registry")
            return False
        
        if model_key not in self.index['models'][commodity]:
            print(f"❌ Model '{model_key}' not found for {commodity}")
            return False
        
        # Update champion in index
        self.index['champions'][commodity] = model_key
        self._save_index()
        
        # Create/update champion.yaml
        spec = self.get_model(commodity, model_name, version)
        if spec:
            spec.is_champion = True
            champion_path = self.models_path / commodity / 'champion.yaml'
            with open(champion_path, 'w') as f:
                f.write(spec.to_yaml())
        
        print(f"✓ Set champion for {commodity}: {model_key}")
        
        return True
    
    def get_champion(self, commodity: str) -> Optional[ModelSpec]:
        """Get the champion model for a commodity."""
        if commodity not in self.index['champions']:
            return None
        
        model_key = self.index['champions'][commodity]
        parts = model_key.rsplit('_', 1)
        
        if len(parts) == 2:
            return self.get_model(commodity, parts[0], parts[1])
        
        return None
    
    def select_best(
        self,
        commodity: str,
        metric: str = 'dir_accuracy',
        higher_is_better: bool = True,
        set_as_champion: bool = False
    ) -> Optional[ModelSpec]:
        """
        Select the best model for a commodity based on a metric.
        
        Args:
            commodity: Commodity name
            metric: Metric to use for selection
            higher_is_better: If True, higher metric values are better
            set_as_champion: If True, set the best model as champion
        
        Returns:
            Best ModelSpec or None
        """
        if commodity not in self.index['models']:
            print(f"❌ Commodity '{commodity}' not found in registry")
            return None
        
        models = self.index['models'][commodity]
        
        best_model = None
        best_value = None
        
        for model_key, model_info in models.items():
            metrics = model_info.get('metrics_summary', {})
            
            # Check for metric in summary or load full spec
            if metric in metrics:
                value = metrics[metric]
            else:
                # Load full spec to check all metrics
                parts = model_key.rsplit('_', 1)
                if len(parts) == 2:
                    spec = self.get_model(commodity, parts[0], parts[1])
                    if spec and metric in spec.metrics:
                        value = spec.metrics[metric]
                    else:
                        continue
                else:
                    continue
            
            if best_value is None:
                best_value = value
                best_model = model_key
            elif higher_is_better and value > best_value:
                best_value = value
                best_model = model_key
            elif not higher_is_better and value < best_value:
                best_value = value
                best_model = model_key
        
        if best_model:
            parts = best_model.rsplit('_', 1)
            spec = self.get_model(commodity, parts[0], parts[1])
            
            print(f"✓ Best model for {commodity} by {metric}: {best_model} ({best_value:.4f})")
            
            if set_as_champion and spec:
                self.set_champion(commodity, parts[0], parts[1])
            
            return spec
        
        return None
    
    def export_predictions(
        self,
        predictions_df: 'pd.DataFrame',
        output_path: str = 'reports/oos_predictions.csv'
    ) -> Path:
        """
        Export predictions in standard format.
        
        Expected columns:
            - commodity
            - week_start
            - y_true
            - y_pred_ret
            - y_pred_dir_prob
            - model_name
        
        Args:
            predictions_df: DataFrame with predictions
            output_path: Output path
        
        Returns:
            Path to saved file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Standardize column names
        column_mapping = {
            'actual_return': 'y_true',
            'predicted_return': 'y_pred_ret',
            'predicted_proba': 'y_pred_dir_prob',
            'direction_probability': 'y_pred_dir_prob',
            'week': 'week_start',
            'date': 'week_start'
        }
        
        df = predictions_df.copy()
        df = df.rename(columns=column_mapping)
        
        # Ensure required columns
        required_cols = ['commodity', 'week_start', 'y_true', 'y_pred_ret', 'y_pred_dir_prob', 'model_name']
        
        for col in required_cols:
            if col not in df.columns:
                if col == 'model_name':
                    df[col] = 'unknown'
                else:
                    print(f"⚠️ Missing column: {col}")
        
        # Select and order columns
        output_cols = [c for c in required_cols if c in df.columns]
        df = df[output_cols]
        
        df.to_csv(output_path, index=False)
        print(f"✓ Exported predictions to: {output_path}")
        
        return output_path
    
    def get_summary(self) -> str:
        """Get a summary of the registry."""
        summary = "=" * 60 + "\n"
        summary += "MODEL REGISTRY SUMMARY\n"
        summary += "=" * 60 + "\n\n"
        
        summary += f"Registry path: {self.registry_path}\n"
        summary += f"Created: {self.index.get('created_at', 'Unknown')}\n"
        summary += f"Last updated: {self.index.get('updated_at', 'Never')}\n\n"
        
        total_models = sum(len(m) for m in self.index['models'].values())
        summary += f"Total models: {total_models}\n"
        summary += f"Commodities: {len(self.index['models'])}\n\n"
        
        summary += "Champions:\n"
        summary += "-" * 40 + "\n"
        for commodity, model_key in self.index.get('champions', {}).items():
            summary += f"  {commodity}: {model_key}\n"
        
        if not self.index.get('champions'):
            summary += "  (none set)\n"
        
        summary += "\nModels by Commodity:\n"
        summary += "-" * 40 + "\n"
        for commodity, models in self.index['models'].items():
            summary += f"\n{commodity}:\n"
            for model_key, info in models.items():
                is_champ = " ⭐" if self.index['champions'].get(commodity) == model_key else ""
                metrics_str = ", ".join(f"{k}={v:.3f}" for k, v in info.get('metrics_summary', {}).items())
                summary += f"  - {model_key}{is_champ}\n"
                if metrics_str:
                    summary += f"    {metrics_str}\n"
        
        return summary


# ============================================================================
# Demo
# ============================================================================

if __name__ == '__main__':
    print("="*80)
    print("MODEL REGISTRY DEMO")
    print("="*80)
    
    # Create registry
    registry = ModelRegistry('demo_registry')
    
    # Register some models
    spec1 = registry.register_model(
        model_name='xgb',
        version='v1',
        commodity='Henry_Hub',
        target='both',
        training_data_start='2022-01-01',
        training_data_end='2023-12-31',
        n_training_samples=500,
        feature_list=['feature_1', 'feature_2', 'feature_3'],
        metrics={
            'mae': 0.025,
            'rmse': 0.032,
            'r2': 0.15,
            'accuracy': 0.56,
            'roc_auc': 0.58,
            'brier_score': 0.24
        },
        artifact_paths={
            'model': 'artifacts/henry_hub_xgb_v1.joblib'
        },
        notes='Initial XGBoost model'
    )
    
    spec2 = registry.register_model(
        model_name='xgb',
        version='v2',
        commodity='Henry_Hub',
        target='both',
        training_data_start='2022-01-01',
        training_data_end='2024-06-30',
        n_training_samples=650,
        feature_list=['feature_1', 'feature_2', 'feature_3', 'feature_4'],
        metrics={
            'mae': 0.022,
            'rmse': 0.029,
            'r2': 0.20,
            'accuracy': 0.59,
            'roc_auc': 0.62,
            'brier_score': 0.22
        },
        artifact_paths={
            'model': 'artifacts/henry_hub_xgb_v2.joblib'
        },
        notes='Improved XGBoost with more features'
    )
    
    # Select best model
    print("\n" + "-"*40)
    best = registry.select_best('Henry_Hub', metric='accuracy', set_as_champion=True)
    
    # Print summary
    print("\n" + registry.get_summary())
    
    # Clean up demo
    import shutil
    shutil.rmtree('demo_registry', ignore_errors=True)
    
    print("✅ Demo complete!")
