"""Simple test to identify the exact issue"""

import pandas as pd
import numpy as np

# Create sample data similar to what the pipeline uses
data = {
    'channelId': ['ch1', 'ch1', 'ch2', 'ch2', 'ch3'],
    'date': pd.date_range('2024-01-01', periods=5, freq='H'),
    'country': ['US', 'US', 'UK', 'UK', 'FR'],
    'device': ['mobile', 'desktop', 'mobile', 'mobile', 'desktop'],
    'browser': ['chrome', 'firefox', 'chrome', 'chrome', 'safari']
}

df = pd.DataFrame(data)

# Test the groupby operations that are failing
print("Test 1: Hourly patterns groupby")
try:
    df['hour'] = df['date'].dt.hour
    hourly_patterns = df.groupby(['channelId', 'hour']).size().unstack(fill_value=0)
    print("Hourly patterns index:", hourly_patterns.index)
    print("Hourly patterns index name:", hourly_patterns.index.name)
    print("Hourly patterns columns:", hourly_patterns.columns)
except Exception as e:
    print(f"Error: {e}")

print("\nTest 2: Country patterns groupby")
try:
    country_patterns = df.groupby(['channelId', 'country']).size().unstack(fill_value=0)
    print("Country patterns index:", country_patterns.index)
    print("Country patterns index name:", country_patterns.index.name)
except Exception as e:
    print(f"Error: {e}")

print("\nTest 3: Device/Browser groupby")
try:
    device_combo = df.groupby(['channelId', 'device', 'browser']).size().unstack(fill_value=0)
    print("Device combo type:", type(device_combo))
    print("Device combo index:", device_combo.index)
    print("Device combo columns:", device_combo.columns)
    
    # Try to flatten
    device_combo_flat = device_combo.unstack(fill_value=0)
    print("\nFlattened device combo columns:", device_combo_flat.columns)
except Exception as e:
    print(f"Error: {e}")

print("\nTest 4: Creating DataFrame with index")
try:
    test_df = pd.DataFrame({
        'channelId': hourly_patterns.index,
        'test_value': [1, 2, 3]
    })
    print("Success creating DataFrame")
except Exception as e:
    print(f"Error creating DataFrame: {e}")