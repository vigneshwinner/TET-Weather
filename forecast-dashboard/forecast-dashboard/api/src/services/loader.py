"""
Data Loader Service
Handles loading and caching of data from files and databases.
"""

import os
import json
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Loads and caches forecast data, SSI values, and backtest results.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.data_dir = Path(config.get("data_dir", "data"))
        self.cache = {}
        self.cache_ttl = config.get("cache_ttl_seconds", 300)  # 5 min default
        self._last_cache_update = {}
        
        # Supported commodities - customize based on your data
        self._commodities = config.get("commodities", [
            "NG",      # Natural Gas
            "CL",      # Crude Oil
            "HO",      # Heating Oil
            "RB",      # RBOB Gasoline
            "ERCOT",   # ERCOT Power
        ])
    
    def _is_cache_valid(self, key: str) -> bool:
        """Check if cache entry is still valid"""
        if key not in self._last_cache_update:
            return False
        elapsed = (datetime.now() - self._last_cache_update[key]).total_seconds()
        return elapsed < self.cache_ttl
    
    def _set_cache(self, key: str, value: Any):
        """Set cache entry with timestamp"""
        self.cache[key] = value
        self._last_cache_update[key] = datetime.now()
    
    def get_commodities(self) -> List[str]:
        """Return list of supported commodity tickers"""
        return self._commodities
    
    def get_ssi(
        self, 
        commodity: str, 
        start: datetime, 
        end: datetime
    ) -> List[Dict[str, Any]]:
        """
        Load SSI (Signal Strength Index) time series for a commodity.
        
        Returns list of {date, ssi_value, components} dicts.
        """
        cache_key = f"ssi_{commodity}"
        
        # Try to load from cache
        if self._is_cache_valid(cache_key):
            df = self.cache[cache_key]
        else:
            # Load from file
            ssi_file = self.data_dir / "ssi" / f"{commodity.lower()}_ssi.csv"
            
            if ssi_file.exists():
                df = pd.read_csv(ssi_file, parse_dates=["date"])
                self._set_cache(cache_key, df)
            else:
                # Return mock data if file doesn't exist (for development)
                logger.warning(f"SSI file not found: {ssi_file}, using mock data")
                return self._generate_mock_ssi(commodity, start, end)
        
        # Filter by date range
        mask = (df["date"] >= start) & (df["date"] <= end)
        filtered = df[mask]
        
        # Format response
        result = []
        for _, row in filtered.iterrows():
            entry = {
                "date": row["date"].strftime("%Y-%m-%d"),
                "ssi_value": float(row.get("ssi_value", row.get("ssi", 0))),
            }
            
            # Include component breakdown if available
            component_cols = [c for c in row.index if c.startswith("component_")]
            if component_cols:
                entry["components"] = {
                    col.replace("component_", ""): float(row[col])
                    for col in component_cols
                }
            
            result.append(entry)
        
        return result
    
    def get_backtest_summary(self, commodity: str) -> Dict[str, Any]:
        """
        Load backtest summary metrics and recent equity curve.
        
        Returns:
            metrics: Dict of performance metrics
            equity_curve: List of {date, value} points
        """
        cache_key = f"backtest_{commodity}"
        
        if self._is_cache_valid(cache_key):
            return self.cache[cache_key]
        
        # Try loading from JSON summary file
        summary_file = self.data_dir / "backtest" / f"{commodity.lower()}_summary.json"
        
        if summary_file.exists():
            with open(summary_file, "r") as f:
                summary = json.load(f)
            self._set_cache(cache_key, summary)
            return summary
        
        # Try loading from CSV equity curve
        equity_file = self.data_dir / "backtest" / f"{commodity.lower()}_equity.csv"
        
        if equity_file.exists():
            df = pd.read_csv(equity_file, parse_dates=["date"])
            
            # Calculate metrics from equity curve
            returns = df["value"].pct_change().dropna()
            
            metrics = {
                "total_return": float((df["value"].iloc[-1] / df["value"].iloc[0] - 1) * 100),
                "sharpe_ratio": float(returns.mean() / returns.std() * (252 ** 0.5)) if returns.std() > 0 else 0,
                "max_drawdown": float(self._calculate_max_drawdown(df["value"])),
                "win_rate": float((returns > 0).mean() * 100),
                "volatility": float(returns.std() * (252 ** 0.5) * 100),
                "num_trades": len(returns),
            }
            
            # Get recent equity points (last 52 weeks)
            recent = df.tail(52)
            equity_curve = [
                {"date": row["date"].strftime("%Y-%m-%d"), "value": float(row["value"])}
                for _, row in recent.iterrows()
            ]
            
            summary = {
                "metrics": metrics,
                "equity_curve": equity_curve,
                "last_updated": datetime.now().isoformat()
            }
            
            self._set_cache(cache_key, summary)
            return summary
        
        # Return mock data for development
        logger.warning(f"Backtest files not found for {commodity}, using mock data")
        return self._generate_mock_backtest(commodity)
    
    def _calculate_max_drawdown(self, equity: pd.Series) -> float:
        """Calculate maximum drawdown percentage"""
        peak = equity.expanding().max()
        drawdown = (equity - peak) / peak
        return float(drawdown.min() * 100)
    
    def _generate_mock_ssi(
        self, 
        commodity: str, 
        start: datetime, 
        end: datetime
    ) -> List[Dict[str, Any]]:
        """Generate mock SSI data for development"""
        import numpy as np
        
        dates = pd.date_range(start, end, freq="W-MON")
        np.random.seed(hash(commodity) % 2**32)
        
        # Generate random walk SSI
        ssi_values = np.cumsum(np.random.randn(len(dates)) * 0.1)
        ssi_values = (ssi_values - ssi_values.min()) / (ssi_values.max() - ssi_values.min())
        ssi_values = ssi_values * 2 - 1  # Scale to [-1, 1]
        
        return [
            {
                "date": d.strftime("%Y-%m-%d"),
                "ssi_value": float(v),
                "components": {
                    "momentum": float(np.random.uniform(-0.5, 0.5)),
                    "mean_reversion": float(np.random.uniform(-0.5, 0.5)),
                    "volatility": float(np.random.uniform(0, 0.5)),
                }
            }
            for d, v in zip(dates, ssi_values)
        ]
    
    def _generate_mock_backtest(self, commodity: str) -> Dict[str, Any]:
        """Generate mock backtest data for development"""
        import numpy as np
        
        np.random.seed(hash(commodity) % 2**32)
        
        # Generate mock equity curve
        dates = pd.date_range(end=datetime.now(), periods=52, freq="W-MON")
        returns = np.random.randn(52) * 0.02 + 0.001  # Slight positive drift
        equity = 100 * np.cumprod(1 + returns)
        
        equity_curve = [
            {"date": d.strftime("%Y-%m-%d"), "value": float(v)}
            for d, v in zip(dates, equity)
        ]
        
        return {
            "metrics": {
                "total_return": float((equity[-1] / equity[0] - 1) * 100),
                "sharpe_ratio": float(np.mean(returns) / np.std(returns) * np.sqrt(52)),
                "max_drawdown": float(np.random.uniform(-15, -5)),
                "win_rate": float(np.random.uniform(50, 60)),
                "volatility": float(np.std(returns) * np.sqrt(52) * 100),
                "num_trades": 52,
            },
            "equity_curve": equity_curve,
            "last_updated": datetime.now().isoformat()
        }
    
    def load_features(self, commodity: str, date: datetime) -> Optional[pd.DataFrame]:
        """
        Load feature data for model inference.
        
        Returns DataFrame with features for the given date.
        """
        features_file = self.data_dir / "features" / f"{commodity.lower()}_features.csv"
        
        if not features_file.exists():
            logger.warning(f"Features file not found: {features_file}")
            return None
        
        df = pd.read_csv(features_file, parse_dates=["date"])
        
        # Get most recent features before the target date
        mask = df["date"] <= date
        if not mask.any():
            return None
        
        return df[mask].tail(1)
