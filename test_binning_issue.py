"""
Test script to reproduce the binning issue with small datasets
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Create small test data that could cause binning issues
small_data = pd.DataFrame({
    'quality_score': [5.0, 5.0, 5.0],  # All same values
    'bot_rate': [0.1, 0.1, 0.1],      # All same values  
    'volume': [100, 100, 100]         # All same values
})

# Test various binning operations
print("Testing binning operations with small identical data...")

# Test 1: plt.hist with fixed bins
try:
    plt.figure(figsize=(8, 6))
    plt.hist(small_data['quality_score'], bins=20)
    print("✓ plt.hist with bins=20 works")
    plt.close()
except Exception as e:
    print(f"✗ plt.hist failed: {e}")
    plt.close()

# Test 2: pd.cut with fixed bins
try:
    bins = [0, 0.1, 0.3, 0.5, 0.7, 1.0]
    result = pd.cut(small_data['bot_rate'], bins=bins)
    print("✓ pd.cut with predefined bins works")
except Exception as e:
    print(f"✗ pd.cut failed: {e}")

# Test 3: pd.qcut (this often fails with identical values)
try:
    result = pd.qcut(small_data['quality_score'], q=5)
    print("✓ pd.qcut with q=5 works")
except Exception as e:
    print(f"✗ pd.qcut failed: {e}")

# Test with edge case: single unique value
single_value_data = pd.DataFrame({
    'quality_score': [3.0, 3.0, 3.0, 3.0, 3.0],
    'bot_rate': [0.2, 0.2, 0.2, 0.2, 0.2],
    'volume': [50, 50, 50, 50, 50]
})

print("\nTesting with single unique values...")

# Test pd.qcut with single value
try:
    result = pd.qcut(single_value_data['quality_score'], q=5)
    print("✓ pd.qcut with single value works")
except Exception as e:
    print(f"✗ pd.qcut with single value failed: {e}")

# Test with very small range
small_range_data = pd.DataFrame({
    'quality_score': [5.0, 5.001, 5.002],
    'volume': [100, 101, 102]
})

print("\nTesting with very small range...")
try:
    result = pd.qcut(small_range_data['quality_score'], q=5)
    print("✓ pd.qcut with small range works")
except Exception as e:
    print(f"✗ pd.qcut with small range failed: {e}")