#!/usr/bin/env python
"""
Register Model CLI
Add entries to the model registry.

Usage:
    python register_model.py --commodity Henry_Hub --model xgb --version v1 \
        --metrics-file results/metrics.json --artifact model.joblib

Issue #17: Unified Evaluation and Model Registry
"""

import argparse
import json
import yaml
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from model_registry import ModelRegistry


def parse_args():
    parser = argparse.ArgumentParser(
        description='Register a trained model in the registry',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Register with metrics file
  python register_model.py --commodity Henry_Hub --model xgb --version v1 \\
      --metrics-file results/xgb_metrics.json \\
      --artifact model.joblib

  # Register with inline metrics
  python register_model.py --commodity Brent --model ridge --version v1 \\
      --metric mae=0.025 --metric accuracy=0.58 \\
      --artifact ridge_model.pkl

  # Full registration
  python register_model.py --commodity Henry_Hub --model xgb --version v2 \\
      --metrics-file results/metrics.json \\
      --artifact reg_model.joblib \\
      --artifact clf_model.joblib \\
      --features-file features.txt \\
      --train-start 2022-01-01 --train-end 2024-06-30 \\
      --notes "Improved model with degree days"
        """
    )
    
    # Required arguments
    parser.add_argument('--commodity', required=True,
                        help='Commodity name (e.g., Henry_Hub, Brent)')
    parser.add_argument('--model', required=True,
                        help='Model name (e.g., xgb, ridge)')
    parser.add_argument('--version', required=True,
                        help='Model version (e.g., v1, v1.1)')
    
    # Metrics (either file or inline)
    parser.add_argument('--metrics-file', type=Path,
                        help='Path to JSON/YAML file with metrics')
    parser.add_argument('--metric', action='append', dest='metrics',
                        help='Inline metric in format key=value (can repeat)')
    
    # Artifacts
    parser.add_argument('--artifact', action='append', dest='artifacts',
                        help='Path to artifact file (can repeat)')
    
    # Training info
    parser.add_argument('--train-start', 
                        help='Training data start date (YYYY-MM-DD)')
    parser.add_argument('--train-end',
                        help='Training data end date (YYYY-MM-DD)')
    parser.add_argument('--n-samples', type=int, default=0,
                        help='Number of training samples')
    
    # Features
    parser.add_argument('--features-file', type=Path,
                        help='Path to file with feature list (one per line)')
    parser.add_argument('--feature', action='append', dest='features',
                        help='Feature name (can repeat)')
    
    # Target
    parser.add_argument('--target', default='both',
                        choices=['magnitude', 'direction', 'both'],
                        help='Target type')
    
    # Metadata
    parser.add_argument('--notes', help='Optional notes')
    parser.add_argument('--set-champion', action='store_true',
                        help='Set this model as champion after registration')
    
    # Registry
    parser.add_argument('--registry', default='models/registry',
                        help='Path to model registry')
    
    return parser.parse_args()


def load_metrics(args) -> dict:
    """Load metrics from file or inline arguments."""
    metrics = {}
    
    # Load from file
    if args.metrics_file:
        path = args.metrics_file
        if path.exists():
            with open(path, 'r') as f:
                if path.suffix in ['.yaml', '.yml']:
                    metrics = yaml.safe_load(f)
                else:
                    metrics = json.load(f)
            print(f"  Loaded metrics from: {path}")
    
    # Add/override with inline metrics
    if args.metrics:
        for m in args.metrics:
            if '=' in m:
                key, value = m.split('=', 1)
                try:
                    metrics[key] = float(value)
                except ValueError:
                    metrics[key] = value
    
    return metrics


def load_features(args) -> list:
    """Load feature list from file or inline arguments."""
    features = []
    
    # Load from file
    if args.features_file:
        path = args.features_file
        if path.exists():
            with open(path, 'r') as f:
                features = [line.strip() for line in f if line.strip()]
            print(f"  Loaded {len(features)} features from: {path}")
    
    # Add inline features
    if args.features:
        features.extend(args.features)
    
    # Default if empty
    if not features:
        features = ['unknown']
    
    return features


def load_artifacts(args) -> dict:
    """Load artifact paths."""
    artifacts = {}
    
    if args.artifacts:
        for i, artifact in enumerate(args.artifacts):
            path = Path(artifact)
            if path.exists():
                # Use filename stem as key
                key = path.stem.split('_')[0] if '_' in path.stem else f'artifact_{i}'
                artifacts[key] = str(path)
            else:
                print(f"  ⚠️ Artifact not found: {artifact}")
    
    return artifacts


def main():
    args = parse_args()
    
    print("="*60)
    print("REGISTER MODEL")
    print("="*60)
    print(f"\nCommodity: {args.commodity}")
    print(f"Model: {args.model}")
    print(f"Version: {args.version}")
    
    # Load data
    print("\n📂 Loading data...")
    metrics = load_metrics(args)
    features = load_features(args)
    artifacts = load_artifacts(args)
    
    if not metrics:
        print("❌ Error: No metrics provided. Use --metrics-file or --metric")
        sys.exit(1)
    
    print(f"\n📊 Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    
    # Create registry
    registry = ModelRegistry(args.registry)
    
    # Register model
    print(f"\n📝 Registering model...")
    
    spec = registry.register_model(
        model_name=args.model,
        version=args.version,
        commodity=args.commodity,
        target=args.target,
        training_data_start=args.train_start or 'unknown',
        training_data_end=args.train_end or 'unknown',
        n_training_samples=args.n_samples,
        feature_list=features,
        metrics=metrics,
        artifact_paths=artifacts,
        notes=args.notes
    )
    
    # Set as champion if requested
    if args.set_champion:
        registry.set_champion(args.commodity, args.model, args.version)
    
    # Print spec
    print(f"\n📄 Model Spec:")
    print("-"*40)
    print(spec.to_yaml())
    
    print("\n✅ Registration complete!")
    print(f"   Spec saved to: {args.registry}/models/{args.commodity}/{args.model}_{args.version}.yaml")


if __name__ == '__main__':
    main()
