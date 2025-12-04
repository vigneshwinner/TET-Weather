"""
Leakage Validation Checks
Validates no target or future feature leakage in the training pipeline.

Issue #17: Unified Evaluation and Model Registry
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from pathlib import Path
import warnings


class LeakageChecker:
    """
    Validates that training data has no look-ahead bias or target leakage.
    """
    
    def __init__(self, target_offset_weeks: int = 1):
        """
        Args:
            target_offset_weeks: Number of weeks the target is shifted forward
        """
        self.target_offset_weeks = target_offset_weeks
        self.validation_results = []
    
    def check_temporal_leakage(
        self,
        features_df: pd.DataFrame,
        date_col: str = 'date',
        target_date_col: str = 'target_date',
        feature_cols: Optional[List[str]] = None
    ) -> Dict[str, any]:
        """
        Check that all features have timestamps <= target timestamp minus offset.
        
        Assert: all feature timestamps <= target timestamp - 1 week
        
        Args:
            features_df: DataFrame with features and dates
            date_col: Column containing feature date
            target_date_col: Column containing target date (prediction date)
            feature_cols: List of feature columns to check
        
        Returns:
            Dictionary with validation results
        """
        results = {
            'check': 'temporal_leakage',
            'passed': True,
            'violations': [],
            'n_checked': 0
        }
        
        if date_col not in features_df.columns:
            results['passed'] = False
            results['error'] = f"Date column '{date_col}' not found"
            return results
        
        # If no explicit target_date_col, infer from date_col
        if target_date_col not in features_df.columns:
            # Assume target is for next week
            features_df = features_df.copy()
            features_df[target_date_col] = pd.to_datetime(features_df[date_col]) + pd.Timedelta(weeks=self.target_offset_weeks)
        
        feature_dates = pd.to_datetime(features_df[date_col])
        target_dates = pd.to_datetime(features_df[target_date_col])
        
        # Features must be at least 1 week before target
        max_feature_date = target_dates - pd.Timedelta(weeks=self.target_offset_weeks)
        
        # Check for violations
        violations_mask = feature_dates > max_feature_date
        n_violations = violations_mask.sum()
        
        results['n_checked'] = len(features_df)
        
        if n_violations > 0:
            results['passed'] = False
            results['violations'] = features_df[violations_mask][[date_col, target_date_col]].head(10).to_dict('records')
            results['n_violations'] = int(n_violations)
            results['violation_rate'] = float(n_violations / len(features_df))
        
        self.validation_results.append(results)
        return results
    
    def check_target_leakage(
        self,
        df: pd.DataFrame,
        target_col: str,
        feature_cols: List[str],
        correlation_threshold: float = 0.95
    ) -> Dict[str, any]:
        """
        Check for features that are suspiciously correlated with target.
        
        High correlation might indicate:
        - Target encoded in features
        - Future information leakage
        - Derived features from target
        
        Args:
            df: DataFrame with features and target
            target_col: Name of target column
            feature_cols: List of feature column names
            correlation_threshold: Flag features with correlation above this
        
        Returns:
            Dictionary with validation results
        """
        results = {
            'check': 'target_leakage',
            'passed': True,
            'suspicious_features': [],
            'n_checked': len(feature_cols)
        }
        
        if target_col not in df.columns:
            results['passed'] = False
            results['error'] = f"Target column '{target_col}' not found"
            return results
        
        target = df[target_col].values
        
        for col in feature_cols:
            if col not in df.columns:
                continue
            
            feature = df[col].values
            
            # Handle missing values
            valid_mask = ~(np.isnan(feature) | np.isnan(target))
            if valid_mask.sum() < 10:
                continue
            
            # Calculate correlation
            corr = np.corrcoef(feature[valid_mask], target[valid_mask])[0, 1]
            
            if abs(corr) > correlation_threshold:
                results['passed'] = False
                results['suspicious_features'].append({
                    'feature': col,
                    'correlation': float(corr),
                    'abs_correlation': float(abs(corr))
                })
        
        # Sort by absolute correlation
        results['suspicious_features'].sort(key=lambda x: x['abs_correlation'], reverse=True)
        
        self.validation_results.append(results)
        return results
    
    def check_train_test_overlap(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        date_col: str = 'date'
    ) -> Dict[str, any]:
        """
        Check that training and test sets don't have overlapping dates.
        
        Args:
            train_df: Training DataFrame
            test_df: Test DataFrame
            date_col: Date column name
        
        Returns:
            Dictionary with validation results
        """
        results = {
            'check': 'train_test_overlap',
            'passed': True,
            'overlapping_dates': [],
            'train_max_date': None,
            'test_min_date': None
        }
        
        train_dates = pd.to_datetime(train_df[date_col])
        test_dates = pd.to_datetime(test_df[date_col])
        
        train_max = train_dates.max()
        test_min = test_dates.min()
        
        results['train_max_date'] = str(train_max.date())
        results['test_min_date'] = str(test_min.date())
        
        # Check for overlap
        overlap = set(train_dates.dt.date) & set(test_dates.dt.date)
        
        if overlap:
            results['passed'] = False
            results['overlapping_dates'] = sorted([str(d) for d in overlap])
            results['n_overlapping'] = len(overlap)
        
        # Also check temporal order
        if train_max >= test_min:
            results['passed'] = False
            results['temporal_violation'] = True
            results['gap_days'] = int((test_min - train_max).days)
        
        self.validation_results.append(results)
        return results
    
    def check_fold_boundaries(
        self,
        folds: List[Tuple[pd.DataFrame, pd.DataFrame]],
        date_col: str = 'date'
    ) -> Dict[str, any]:
        """
        Validate temporal CV fold boundaries are correct.
        
        Args:
            folds: List of (train_df, test_df) tuples
            date_col: Date column name
        
        Returns:
            Dictionary with validation results
        """
        results = {
            'check': 'fold_boundaries',
            'passed': True,
            'n_folds': len(folds),
            'fold_details': [],
            'violations': []
        }
        
        for i, (train_df, test_df) in enumerate(folds):
            fold_result = self.check_train_test_overlap(train_df, test_df, date_col)
            
            fold_info = {
                'fold': i + 1,
                'train_size': len(train_df),
                'test_size': len(test_df),
                'train_max_date': fold_result['train_max_date'],
                'test_min_date': fold_result['test_min_date'],
                'passed': fold_result['passed']
            }
            
            results['fold_details'].append(fold_info)
            
            if not fold_result['passed']:
                results['passed'] = False
                results['violations'].append({
                    'fold': i + 1,
                    'issue': fold_result
                })
        
        self.validation_results.append(results)
        return results
    
    def check_feature_availability(
        self,
        features_df: pd.DataFrame,
        feature_cols: List[str],
        date_col: str = 'date',
        min_history_days: int = 7
    ) -> Dict[str, any]:
        """
        Check that features would be available at prediction time.
        
        Args:
            features_df: DataFrame with features
            feature_cols: List of feature columns
            date_col: Date column
            min_history_days: Minimum days of history required
        
        Returns:
            Dictionary with validation results
        """
        results = {
            'check': 'feature_availability',
            'passed': True,
            'unavailable_features': [],
            'delayed_features': []
        }
        
        # Check for features that might not be available in real-time
        realtime_indicators = ['_forecast', '_estimate', '_preliminary']
        delayed_indicators = ['_revised', '_final', '_actual']
        
        for col in feature_cols:
            col_lower = col.lower()
            
            # Check for potentially forward-looking names
            if any(ind in col_lower for ind in realtime_indicators):
                results['unavailable_features'].append({
                    'feature': col,
                    'reason': 'Name suggests forecast/estimate data'
                })
            
            # Check for delayed data indicators
            if any(ind in col_lower for ind in delayed_indicators):
                results['delayed_features'].append({
                    'feature': col,
                    'reason': 'Name suggests revised/final data which may be delayed'
                })
        
        if results['unavailable_features']:
            results['passed'] = False
            results['n_unavailable'] = len(results['unavailable_features'])
        
        self.validation_results.append(results)
        return results
    
    def validate_all(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str,
        date_col: str = 'date'
    ) -> Dict[str, any]:
        """
        Run all leakage checks.
        
        Args:
            train_df: Training DataFrame
            test_df: Test DataFrame
            feature_cols: List of feature columns
            target_col: Target column name
            date_col: Date column name
        
        Returns:
            Dictionary with all validation results
        """
        print("\n🔍 Running Leakage Validation Checks...")
        
        results = {
            'overall_passed': True,
            'checks': {}
        }
        
        # 1. Train/test overlap
        print("  Checking train/test overlap...")
        overlap_result = self.check_train_test_overlap(train_df, test_df, date_col)
        results['checks']['train_test_overlap'] = overlap_result
        if not overlap_result['passed']:
            results['overall_passed'] = False
        
        # 2. Target leakage in training data
        print("  Checking target leakage...")
        leakage_result = self.check_target_leakage(train_df, target_col, feature_cols)
        results['checks']['target_leakage'] = leakage_result
        if not leakage_result['passed']:
            results['overall_passed'] = False
        
        # 3. Feature availability
        print("  Checking feature availability...")
        availability_result = self.check_feature_availability(train_df, feature_cols, date_col)
        results['checks']['feature_availability'] = availability_result
        if not availability_result['passed']:
            results['overall_passed'] = False
        
        # Summary
        if results['overall_passed']:
            print("  ✅ All leakage checks passed!")
        else:
            print("  ⚠️  Some leakage checks failed!")
            for check_name, check_result in results['checks'].items():
                if not check_result['passed']:
                    print(f"    ❌ {check_name}")
        
        return results
    
    def get_report(self) -> str:
        """Generate a text report of all validation results."""
        report = "=" * 60 + "\n"
        report += "LEAKAGE VALIDATION REPORT\n"
        report += "=" * 60 + "\n\n"
        
        for result in self.validation_results:
            check_name = result.get('check', 'Unknown')
            passed = result.get('passed', False)
            status = "✅ PASSED" if passed else "❌ FAILED"
            
            report += f"{check_name}: {status}\n"
            report += "-" * 40 + "\n"
            
            for key, value in result.items():
                if key not in ['check', 'passed']:
                    report += f"  {key}: {value}\n"
            
            report += "\n"
        
        return report


# ============================================================================
# Standalone validation functions
# ============================================================================

def validate_no_leakage(
    features_df: pd.DataFrame,
    target_col: str,
    date_col: str = 'date',
    target_date_col: str = 'week'
) -> bool:
    """
    Quick validation that there's no obvious leakage.
    
    Args:
        features_df: DataFrame with features, target, and dates
        target_col: Target column name
        date_col: Feature date column
        target_date_col: Target/prediction date column
    
    Returns:
        True if no leakage detected, False otherwise
    """
    checker = LeakageChecker()
    
    # Get feature columns
    exclude_cols = [date_col, target_date_col, target_col, 'commodity', 'Date', 'week']
    feature_cols = [c for c in features_df.columns if c not in exclude_cols]
    
    # Check target leakage
    leakage_result = checker.check_target_leakage(
        features_df, target_col, feature_cols, correlation_threshold=0.95
    )
    
    if not leakage_result['passed']:
        print(f"⚠️  Potential target leakage detected!")
        for feat in leakage_result['suspicious_features'][:5]:
            print(f"    {feat['feature']}: correlation = {feat['correlation']:.3f}")
        return False
    
    return True


def assert_temporal_integrity(
    train_dates: pd.Series,
    test_dates: pd.Series,
    gap_weeks: int = 0
) -> None:
    """
    Assert that train dates are strictly before test dates.
    
    Args:
        train_dates: Series of training dates
        test_dates: Series of test dates
        gap_weeks: Minimum gap between train and test (in weeks)
    
    Raises:
        AssertionError if temporal integrity is violated
    """
    train_max = pd.to_datetime(train_dates).max()
    test_min = pd.to_datetime(test_dates).min()
    
    min_gap = pd.Timedelta(weeks=gap_weeks)
    
    assert train_max + min_gap <= test_min, \
        f"Temporal integrity violated: train_max={train_max}, test_min={test_min}, required_gap={gap_weeks} weeks"


# ============================================================================
# Demo
# ============================================================================

if __name__ == '__main__':
    print("="*80)
    print("LEAKAGE CHECKER DEMO")
    print("="*80)
    
    # Create sample data
    np.random.seed(42)
    n = 100
    
    dates = pd.date_range('2023-01-01', periods=n, freq='W')
    
    df = pd.DataFrame({
        'date': dates,
        'feature_1': np.random.randn(n),
        'feature_2': np.random.randn(n),
        'target': np.random.randn(n) * 0.05
    })
    
    # Add a leaky feature (highly correlated with target)
    df['leaky_feature'] = df['target'] * 0.99 + np.random.randn(n) * 0.01
    
    # Split into train/test
    train_df = df.iloc[:80]
    test_df = df.iloc[80:]
    
    # Run checks
    checker = LeakageChecker()
    
    feature_cols = ['feature_1', 'feature_2', 'leaky_feature']
    
    results = checker.validate_all(
        train_df=train_df,
        test_df=test_df,
        feature_cols=feature_cols,
        target_col='target',
        date_col='date'
    )
    
    print("\n" + checker.get_report())
