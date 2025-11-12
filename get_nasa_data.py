import requests
import pandas as pd
from datetime import datetime, timedelta
import time

class NASAPowerFetcher:
    """NASA POWER API data fetcher for weather data - returns DataFrame or dict"""
    
    def __init__(self):
        self.base_url = "https://power.larc.nasa.gov/api/temporal/daily/point"
        
        # Key energy/commodity producing regions
        self.regions = {
            'Houston_US': {'lat': 29.7604, 'lon': -95.3698, 'description': 'US Energy Hub'},
            'Permian_Basin_US': {'lat': 31.8457, 'lon': -102.3676, 'description': 'US Oil Region'},
            'North_Sea_Norway': {'lat': 60.5000, 'lon': 4.0000, 'description': 'North Sea Oil/Gas'},
            'Qatar_LNG': {'lat': 25.3548, 'lon': 51.1839, 'description': 'LNG Production'},
            'Rotterdam_Netherlands': {'lat': 51.9225, 'lon': 4.4792, 'description': 'European Energy Hub'},
            'Singapore': {'lat': 1.3521, 'lon': 103.8198, 'description': 'Asian Trading Hub'},
            'Alberta_Canada': {'lat': 53.5461, 'lon': -113.4938, 'description': 'Canadian Oil Sands'},
            'Cushing_Oklahoma': {'lat': 35.9849, 'lon': -96.7697, 'description': 'US Oil Storage Hub'}
        }
        
        # Weather parameters to fetch
        self.parameters = [
            'T2M',           # Temperature at 2m (°C)
            'T2M_MAX',       # Maximum Temperature (°C)
            'T2M_MIN',       # Minimum Temperature (°C)
            'PRECTOTCORR',   # Precipitation (mm/day)
            'ALLSKY_SFC_SW_DWN',  # Solar Radiation (kW-hr/m²/day)
            'WS10M',         # Wind Speed at 10m (m/s)
            'WD10M'          # Wind Direction at 10m (degrees)
        ]
    
    
    def fetch_data(self, years=3, end_date=None, return_format='dataframe'):
        """
        Fetch weather data for all regions
        
        Args:
            years: Number of years of historical data (default: 3)
            end_date: End date for data (default: yesterday)
            return_format: 'dataframe' or 'dict' (default: 'dataframe')
        
        Returns:
            pandas DataFrame or dict with all weather data
        """
        if end_date is None:
            # Use yesterday as end date (today's data might not be available)
            end_date = datetime.now() - timedelta(days=1)
        else:
            end_date = pd.to_datetime(end_date)
        
        start_date = end_date - timedelta(days=365 * years)
        
        # Format dates for API
        start_str = start_date.strftime('%Y%m%d')
        end_str = end_date.strftime('%Y%m%d')
        
        print(f"Fetching data from {start_str} to {end_str}")
        print(f"Regions: {len(self.regions)}")
        
        all_data = []
        
        for region_name, coords in self.regions.items():
            print(f"Fetching: {region_name}")
            
            # Build API request
            params = {
                'parameters': ','.join(self.parameters),
                'community': 'RE',
                'longitude': coords['lon'],
                'latitude': coords['lat'],
                'start': start_str,
                'end': end_str,
                'format': 'JSON'
            }
            
            try:
                response = requests.get(self.base_url, params=params, timeout=30)
                response.raise_for_status()
                
                data = response.json()
                
                # Extract the actual data
                if 'properties' in data and 'parameter' in data['properties']:
                    param_data = data['properties']['parameter']
                    
                    # Convert to DataFrame
                    df = pd.DataFrame(param_data)
                    
                    # Add metadata
                    df['region'] = region_name
                    df['latitude'] = coords['lat']
                    df['longitude'] = coords['lon']
                    df['description'] = coords['description']
                    
                    # Convert index to datetime
                    df.index = pd.to_datetime(df.index, format='%Y%m%d')
                    df.index.name = 'date'
                    
                    all_data.append(df)
                    print(f"  ✓ Retrieved {len(df)} days")
                else:
                    print(f"  ✗ No data found")
                
                # Rate limiting
                time.sleep(1)
                
            except requests.exceptions.RequestException as e:
                print(f"  ✗ Error: {str(e)[:100]}")
                continue
        
        if all_data:
            # Combine all dataframes
            final_df = pd.concat(all_data)
            
            # Reset index to make date a column
            final_df = final_df.reset_index()
            
            # Rename columns for clarity
            column_names = {
                'T2M': 'temp_avg_c',
                'T2M_MAX': 'temp_max_c',
                'T2M_MIN': 'temp_min_c',
                'PRECTOTCORR': 'precipitation_mm',
                'ALLSKY_SFC_SW_DWN': 'solar_radiation_kwh_m2',
                'WS10M': 'wind_speed_ms',
                'WD10M': 'wind_direction_deg'
            }
            final_df = final_df.rename(columns=column_names)
            
            # Reorder columns
            cols = ['date', 'region', 'description', 'latitude', 'longitude'] + \
                   [col for col in final_df.columns if col not in 
                    ['date', 'region', 'description', 'latitude', 'longitude']]
            final_df = final_df[cols]
            
            # Handle missing values (-999 in NASA POWER means no data)
            final_df = final_df.replace(-999, pd.NA)
            
            print(f"\n✓ Complete! {len(final_df):,} records fetched")
            
            # Return in requested format
            if return_format == 'dict':
                return self.to_dict(final_df)
            else:
                return final_df
        else:
            print("No data retrieved")
            return pd.DataFrame() if return_format == 'dataframe' else {}
    
    def to_dict(self, df):
        """
        Convert DataFrame to dictionary format
        
        Args:
            df: pandas DataFrame
            
        Returns:
            dict with data and metadata
        """
        if df.empty:
            return {}
        
        # Convert datetime to string for JSON compatibility
        df_copy = df.copy()
        df_copy['date'] = df_copy['date'].dt.strftime('%Y-%m-%d')
        
        return {
            'metadata': {
                'total_records': len(df_copy),
                'date_range': {
                    'start': df_copy['date'].min(),
                    'end': df_copy['date'].max()
                },
                'regions': df_copy['region'].unique().tolist(),
                'parameters': [col for col in df_copy.columns 
                             if col not in ['date', 'region', 'description', 'latitude', 'longitude']]
            },
            'data': df_copy.to_dict('records')
        }


def get_weather_data(years=3, return_format='dataframe'):
    """
    Quick function to get weather data
    
    Args:
        years: Number of years of data (default: 3)
        return_format: 'dataframe' or 'dict'
    
    Returns:
        Weather data in specified format
    """
    fetcher = NASAPowerFetcher()
    return fetcher.fetch_data(years=years, return_format=return_format)


# Example usage
if __name__ == "__main__":
    # Quick test - fetch data as DataFrame
    print("Testing NASA POWER API fetcher...")
    df = get_weather_data(years=3, return_format='dataframe')
    
    if not df.empty:
        print(f"\nDataFrame shape: {df.shape}")
        print("\nFirst 5 rows:")
        print(df.head())
    
    # Or get as dictionary/JSON format
    # data_dict = get_weather_data(years=3, return_format='dict')
    # print(data_dict['metadata'])