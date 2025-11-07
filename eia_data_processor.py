#!/usr/bin/env python3

import pandas as pd
import requests

class EIADataFetcher:
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.eia.gov/v2/seriesid/"
        
        # Series IDs for crude, natural gas, gasoline, distillate, propane
        self.series = {
            # Crude Oil
            "crude_inventory": "PET.WCESTUS1.W",
            "crude_production": "PET.WCRFPUS2.W",
            "crude_imports": "PET.WCEIMUS2.W",
            "refinery_utilization": "PET.WPULEUS3.W",
            
            # Natural Gas
            "natgas_inventory": "NG.NW2_EPG0_SWO_R48_BCF.W",
            # Note: natgas_net_change will be calculated from inventory changes
            
            # Gasoline  
            "gasoline_inventory": "PET.WGTSTUS1.W",
            "gasoline_production": "PET.WGFUPUS2.W",
            
            # Distillate (diesel/heating oil - weather sensitive)
            "distillate_inventory": "PET.WDISTUS1.W",
            "distillate_production": "PET.WDIUPUS2.W",
            
            # Propane (very weather sensitive - winter heating)
            "propane_inventory": "PET.WPRSTUS1.W",
        }
    
    def fetch_series(self, series_id: str) -> pd.DataFrame:
        """Fetch a single series from EIA API."""
        url = f"{self.base_url}{series_id}"
        resp = requests.get(url, params={"api_key": self.api_key}, timeout=30)
        resp.raise_for_status()
        
        data = resp.json()["response"]["data"]
        df = pd.DataFrame(data)[["period", "value"]]
        df.columns = ["date", "value"]
        return df
    
    def fetch_all(self) -> pd.DataFrame:
        """Fetch all series and merge into single DataFrame."""
        
        frames = []
        for name, series_id in self.series.items():
            print(f"Fetching {name}...")
            try:
                df = self.fetch_series(series_id)
                df = df.rename(columns={"value": name})
                frames.append(df)
                print(f"  ✓ Success")
            except Exception as e:
                print(f"  ✗ Failed: {e}")
                continue
        
        if not frames:
            raise RuntimeError("No data series fetched successfully")
        
        # Merge all on date
        result = frames[0]
        for df in frames[1:]:
            result = result.merge(df, on="date", how="outer")
        
        # Standardize time format
        result["date"] = pd.to_datetime(result["date"])
        result = result.sort_values("date").reset_index(drop=True)
        
        # Convert to numeric
        for col in result.columns:
            if col != "date":
                result[col] = pd.to_numeric(result[col], errors="coerce")
        
        # Calculate natural gas net change (weekly injection/withdrawal)
        if "natgas_inventory" in result.columns:
            result["natgas_net_change"] = result["natgas_inventory"].diff()
        
        print(f"\nFetched {len(result)} weekly records")
        print(f"Date range: {result['date'].min().date()} to {result['date'].max().date()}")
        print(f"Columns: {list(result.columns)}")
        
        return result


if __name__ == "__main__":
    # Example usage
    api_key = 'JMlLALgGbXN9BT2khJUocOZzsuJsdGTACakEAEn8'
    fetcher = EIADataFetcher(api_key)
    
    df = fetcher.fetch_all()
    
    print("\nLatest 5 weeks:")
    print(df[['date', 'crude_inventory', 'natgas_inventory', 'gasoline_inventory', 
              'distillate_inventory', 'propane_inventory']].tail())
    
    print("\nAll columns:")
    print(list(df.columns))
    
    print("\nData ready for weather correlation analysis!")
    print("Key weather-sensitive series:")
    print("  - natgas_net_change: Weekly storage injections/withdrawals (calculated)")
    print("  - distillate_inventory: Heating oil (winter demand)")
    print("  - propane_inventory: Very sensitive to cold snaps")
    print("  - gasoline_production: Summer driving season indicator")