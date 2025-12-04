#!/usr/bin/env python
"""
Select Best Model CLI
Pick the champion per target and commodity by chosen metric.

Usage:
    python select_best.py --commodity Henry_Hub --metric accuracy
    python select_best.py --all --metric roc_auc --set-champion

Issue #17: Unified Evaluation and Model Registry
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from model_registry import ModelRegistry


def parse_args():
    parser = argparse.ArgumentParser(
        description='Select the best model for a commodity based on a metric',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Find best model for a specific commodity
  python select_best.py --commodity Henry_Hub --metric accuracy

  # Find and set champion for all commodities
  python select_best.py --all --metric roc_auc --set-champion

  # Find best by lower-is-better metric (e.g., Brier score)
  python select_best.py --commodity Brent --metric brier_score --lower-is-better

  # Compare all models
  python select_best.py --all --compare

Available metrics:
  Magnitude: mae, rmse, r2, mape
  Direction: accuracy, f1, roc_auc, pr_auc, brier_score, log_loss
        """
    )
    
    # Scope
    parser.add_argument('--commodity',
                        help='Specific commodity to evaluate')
    parser.add_argument('--all', action='store_true',
                        help='Evaluate all commodities')
    
    # Metric selection
    parser.add_argument('--metric', default='accuracy',
                        help='Metric to use for selection (default: accuracy)')
    parser.add_argument('--lower-is-better', action='store_true',
                        help='If set, lower metric values are better')
    
    # Actions
    parser.add_argument('--set-champion', action='store_true',
                        help='Set selected model as champion')
    parser.add_argument('--compare', action='store_true',
                        help='Show comparison table of all models')
    
    # Registry
    parser.add_argument('--registry', default='models/registry',
                        help='Path to model registry')
    
    # Output
    parser.add_argument('--output', type=Path,
                        help='Save results to file (JSON or YAML)')
    
    return parser.parse_args()


def print_model_comparison(registry: ModelRegistry, commodity: Optional[str] = None):
    """Print a comparison table of all models."""
    models = registry.list_models(commodity)
    
    if not models:
        print("No models found in registry.")
        return
    
    # Group by commodity
    by_commodity = {}
    for m in models:
        comm = m['commodity']
        if comm not in by_commodity:
            by_commodity[comm] = []
        by_commodity[comm].append(m)
    
    print("\n" + "="*80)
    print("MODEL COMPARISON")
    print("="*80)
    
    for comm, comm_models in by_commodity.items():
        print(f"\n📊 {comm}")
        print("-"*70)
        
        # Header
        print(f"{'Model':<20} {'MAE':<10} {'R²':<10} {'Acc':<10} {'AUC':<10} {'Brier':<10} {'Champion':<8}")
        print("-"*70)
        
        for m in comm_models:
            metrics = m.get('metrics', {})
            champ = "⭐" if m.get('is_champion') else ""
            
            mae = f"{metrics.get('mae', 0):.4f}" if 'mae' in metrics else "N/A"
            r2 = f"{metrics.get('r2', 0):.4f}" if 'r2' in metrics else "N/A"
            acc = f"{metrics.get('accuracy', 0):.4f}" if 'accuracy' in metrics else "N/A"
            auc = f"{metrics.get('roc_auc', 0):.4f}" if 'roc_auc' in metrics else "N/A"
            brier = f"{metrics.get('brier_score', 0):.4f}" if 'brier_score' in metrics else "N/A"
            
            print(f"{m['model_key']:<20} {mae:<10} {r2:<10} {acc:<10} {auc:<10} {brier:<10} {champ:<8}")


def select_best_for_commodity(
    registry: ModelRegistry,
    commodity: str,
    metric: str,
    higher_is_better: bool,
    set_champion: bool
) -> Optional[dict]:
    """Select best model for a commodity."""
    
    print(f"\n🔍 Selecting best model for {commodity} by {metric}...")
    
    best = registry.select_best(
        commodity=commodity,
        metric=metric,
        higher_is_better=higher_is_better,
        set_as_champion=set_champion
    )
    
    if best:
        return {
            'commodity': commodity,
            'model': f"{best.model_name}_{best.version}",
            'metric': metric,
            'value': best.metrics.get(metric),
            'is_champion': set_champion
        }
    else:
        print(f"  ⚠️ No models found for {commodity}")
        return None


def main():
    args = parse_args()
    
    print("="*60)
    print("SELECT BEST MODEL")
    print("="*60)
    
    # Create registry
    registry = ModelRegistry(args.registry)
    
    # Determine direction
    higher_is_better = not args.lower_is_better
    direction = "higher" if higher_is_better else "lower"
    print(f"\nMetric: {args.metric} ({direction} is better)")
    
    # Comparison mode
    if args.compare:
        print_model_comparison(registry, args.commodity)
        return
    
    # Get commodities to evaluate
    if args.commodity:
        commodities = [args.commodity]
    elif args.all:
        commodities = list(registry.index['models'].keys())
    else:
        print("❌ Error: Specify --commodity or --all")
        sys.exit(1)
    
    if not commodities:
        print("No commodities found in registry.")
        sys.exit(0)
    
    # Select best for each commodity
    results = []
    
    for commodity in commodities:
        result = select_best_for_commodity(
            registry=registry,
            commodity=commodity,
            metric=args.metric,
            higher_is_better=higher_is_better,
            set_champion=args.set_champion
        )
        if result:
            results.append(result)
    
    # Summary
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    
    if results:
        print(f"\n{'Commodity':<20} {'Best Model':<25} {args.metric:<15} {'Champion':<10}")
        print("-"*70)
        
        for r in results:
            value = f"{r['value']:.4f}" if r['value'] is not None else "N/A"
            champ = "✓" if r['is_champion'] else ""
            print(f"{r['commodity']:<20} {r['model']:<25} {value:<15} {champ:<10}")
    else:
        print("No results found.")
    
    # Save output
    if args.output:
        import json
        import yaml
        
        output_data = {
            'metric': args.metric,
            'higher_is_better': higher_is_better,
            'results': results
        }
        
        with open(args.output, 'w') as f:
            if args.output.suffix in ['.yaml', '.yml']:
                yaml.dump(output_data, f, default_flow_style=False)
            else:
                json.dump(output_data, f, indent=2)
        
        print(f"\n💾 Results saved to: {args.output}")
    
    # Print current champions
    print("\n📊 Current Champions:")
    print("-"*40)
    for commodity, model_key in registry.index.get('champions', {}).items():
        print(f"  {commodity}: {model_key}")
    
    print("\n✅ Selection complete!")


if __name__ == '__main__':
    main()
