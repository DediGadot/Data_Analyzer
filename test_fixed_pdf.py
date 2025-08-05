"""
Test script with small dataset to verify binning fixes
"""

import pandas as pd
import numpy as np
import sys
import os

# Add the current directory to Python path
sys.path.append('/home/fiod/shimshi')

# Create test data that would cause the original binning errors
print("Creating test data with edge cases...")

# Test case 1: Very small dataset with identical values
small_identical_df = pd.DataFrame({
    'channelId': ['CH001', 'CH002', 'CH003'],
    'quality_score': [5.0, 5.0, 5.0],  # All identical
    'bot_rate': [0.2, 0.2, 0.2],      # All identical
    'volume': [100, 100, 100],        # All identical
    'quality_category': ['Medium-High', 'Medium-High', 'Medium-High'],
    'high_risk': [False, False, False]
})

# Test case 2: Small dataset with very limited unique values
small_limited_df = pd.DataFrame({
    'channelId': ['CH001', 'CH002', 'CH003', 'CH004', 'CH005'],
    'quality_score': [3.0, 3.0, 7.0, 7.0, 7.0],  # Only 2 unique values
    'bot_rate': [0.1, 0.1, 0.8, 0.8, 0.8],       # Only 2 unique values
    'volume': [50, 50, 200, 200, 200],           # Only 2 unique values
    'quality_category': ['Low', 'Low', 'High', 'High', 'High'],
    'high_risk': [False, False, True, True, True]
})

# Test case 3: Single row dataset
single_row_df = pd.DataFrame({
    'channelId': ['CH001'],
    'quality_score': [4.5],
    'bot_rate': [0.3],
    'volume': [150],
    'quality_category': ['Medium-High'],
    'high_risk': [False]
})

test_cases = [
    ("Small Identical Data", small_identical_df),
    ("Small Limited Data", small_limited_df), 
    ("Single Row Data", single_row_df)
]

try:
    from pdf_report_generator_multilingual import MultilingualPDFReportGenerator
    print("✓ Successfully imported MultilingualPDFReportGenerator")
    
    # Test each case
    for test_name, quality_df in test_cases:
        print(f"\n=== Testing {test_name} ===")
        print(f"Data shape: {quality_df.shape}")
        print(f"Unique quality scores: {quality_df['quality_score'].nunique()}")
        print(f"Unique bot rates: {quality_df['bot_rate'].nunique()}")
        print(f"Unique volumes: {quality_df['volume'].nunique()}")
        
        # Create minimal anomaly data
        anomaly_df = pd.DataFrame({
            'channelId': quality_df['channelId'].iloc[:min(2, len(quality_df))],
            'temporal_anomaly': [True, False][:min(2, len(quality_df))],
            'geographic_anomaly': [False, True][:min(2, len(quality_df))],
        })
        anomaly_cols = [col for col in anomaly_df.columns if 'anomaly' in col]
        anomaly_df['overall_anomaly_count'] = anomaly_df[anomaly_cols].sum(axis=1)
        
        # Mock results
        final_results = {'test': 'data'}
        pipeline_results = {'test': 'data'}
        
        try:
            generator = MultilingualPDFReportGenerator()
            
            # Test individual plotting methods that were problematic
            print("  Testing quality distribution plot...")
            plot_path = generator.create_quality_distribution_plot(quality_df, 'en')
            print(f"  ✓ Quality distribution plot created: {os.path.basename(plot_path)}")
            
            print("  Testing risk matrix plot...")
            plot_path = generator.create_risk_matrix(quality_df, 'en')
            print(f"  ✓ Risk matrix plot created: {os.path.basename(plot_path)}")
            
            print("  Testing bot rate boxplot...")
            plot_path = generator.create_bot_rate_boxplot(quality_df, 'en')
            print(f"  ✓ Bot rate boxplot created: {os.path.basename(plot_path)}")
            
            print("  Testing cluster visualization...")
            plot_path = generator.create_cluster_visualization(quality_df, 'en')
            print(f"  ✓ Cluster visualization created: {os.path.basename(plot_path)}")
            
            print("  Testing cluster quality chart...")
            plot_path = generator.create_cluster_quality_chart(quality_df, 'en')
            print(f"  ✓ Cluster quality chart created: {os.path.basename(plot_path)}")
            
            print("  Testing quality trend plot...")
            plot_path = generator.create_quality_trend_plot(quality_df, 'en')
            print(f"  ✓ Quality trend plot created: {os.path.basename(plot_path)}")
            
            print(f"  ✅ {test_name} - All plots created successfully!")
            
        except Exception as e:
            print(f"  ❌ {test_name} failed: {e}")
            import traceback
            traceback.print_exc()
            
except ImportError as e:
    print(f"❌ Failed to import: {e}")
    print("This might be due to missing dependencies.")

print("\n=== Test Summary ===")
print("If you see ✅ for all test cases, the binning fixes are working correctly!")
print("The fixes handle:")
print("- Identical values in datasets") 
print("- Very few unique values")
print("- Single-row datasets")
print("- Empty or insufficient data for clustering/binning")