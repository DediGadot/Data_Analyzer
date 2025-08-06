#!/usr/bin/env python3
"""
Simple verification that the column preservation fix works
"""

import pandas as pd
import numpy as np
import sys
import os
sys.path.append('/home/fiod/shimshi')

def verify_column_preservation():
    """Test column preservation by simulating the pipeline steps directly"""
    print("=== DIRECT COLUMN PRESERVATION VERIFICATION ===")
    
    # Create test data with all 30 original columns
    np.random.seed(42)
    n_rows = 50
    
    # Create the exact original column structure
    test_data = {
        'date': pd.date_range('2025-08-01', periods=n_rows, freq='h'),
        'keyword': np.random.choice(['kw1', 'kw2', 'kw3'], n_rows),
        'country': np.random.choice(['US', 'CA', 'UK'], n_rows),
        'browser': np.random.choice(['chrome', 'firefox'], n_rows),
        'device': np.random.choice(['mobile', 'desktop'], n_rows),
        'referrer': np.random.choice(['google.com', 'facebook.com', None], n_rows),
        'ip': [f'192.168.1.{i}' for i in range(n_rows)],
        'publisherId': [f'pub_{i%3}' for i in range(n_rows)],
        'channelId': [f'ch_{i%5}' for i in range(n_rows)],
        'advertiserId': [f'adv_{i%2}' for i in range(n_rows)],
        'feedId': [f'feed_{i%3}' for i in range(n_rows)],
        'browserMajorVersion': np.random.choice([90, 91, 92], n_rows),
        'userId': [f'user_{i%10}' for i in range(n_rows)],
        'isLikelyBot': np.random.choice([True, False], n_rows),
        'ipClassification': np.random.choice(['clean', 'suspicious'], n_rows),
        'isIpDatacenter': np.random.choice([True, False], n_rows),
        'datacenterName': np.random.choice(['AWS', 'Google', None], n_rows),
        'ipHostName': [f'host{i}.com' if i % 3 == 0 else None for i in range(n_rows)],
        'isIpAnonymous': np.random.choice([True, False], n_rows),
        'isIpCrawler': np.random.choice([True, False], n_rows),
        'isIpPublicProxy': np.random.choice([True, False], n_rows),
        'isIpVPN': np.random.choice([True, False], n_rows),
        'isIpHostingService': np.random.choice([True, False], n_rows),
        'isIpTOR': np.random.choice([True, False], n_rows),
        'isIpResidentialProxy': np.random.choice([True, False], n_rows),
        'performance': np.random.uniform(0.5, 2.0, n_rows),
        'detection': np.random.choice(['detected', 'undetected'], n_rows),
        'platform': np.random.choice(['web', 'mobile'], n_rows),
        'location': np.random.choice(['US-CA', 'US-NY'], n_rows),
        'userAgent': [f'Mozilla/5.0 Agent{i}' for i in range(n_rows)]
    }
    
    original_df = pd.DataFrame(test_data)
    print(f"Original data columns: {original_df.shape[1]}")
    print(f"Original data shape: {original_df.shape}")
    
    # Create mock quality results
    quality_results_df = pd.DataFrame({
        'channelId': [f'ch_{i}' for i in range(5)],
        'quality_score': [3.2, 7.5, 5.8, 2.1, 8.3],
        'high_risk': [True, False, False, True, False]
    })
    
    # Create mock anomaly results  
    anomaly_results = pd.DataFrame({
        'channelId': [f'ch_{i}' for i in range(5)],
        'temporal_anomaly': [True, False, True, False, False],
        'geographic_anomaly': [False, True, False, True, False],
        'device_anomaly': [False, False, True, False, True],
        'behavioral_anomaly': [True, False, False, False, True],
        'volume_anomaly': [False, False, False, True, False],
        'overall_anomaly_count': [2, 1, 2, 2, 2]
    })
    
    # Create mock features_df
    features_df = original_df.copy()
    # Add some dummy feature columns
    features_df['feature1'] = np.random.random(n_rows)
    features_df['feature2'] = np.random.random(n_rows)
    
    print(f"Features data shape: {features_df.shape}")
    
    # Test the FraudClassifier directly
    try:
        from fraud_classifier import FraudClassifier
        
        classifier = FraudClassifier()
        
        print(f"\nTesting FraudClassifier.classify_dataset()...")
        print(f"Input original_df columns: {len(original_df.columns)}")
        
        # This should preserve all original columns
        classified_df = classifier.classify_dataset(
            original_df, quality_results_df, anomaly_results, features_df
        )
        
        print(f"Output classified_df columns: {len(classified_df.columns)}")
        print(f"Output classified_df shape: {classified_df.shape}")
        
        # Check which original columns are preserved
        original_cols = set(original_df.columns)
        classified_cols = set(classified_df.columns)
        
        preserved_original = original_cols.intersection(classified_cols)
        missing_original = original_cols - classified_cols
        new_classification_cols = classified_cols - original_cols
        
        print(f"\n=== COLUMN ANALYSIS ===")
        print(f"Original columns: {len(original_cols)}")
        print(f"Preserved original columns: {len(preserved_original)}")
        print(f"Missing original columns: {len(missing_original)}")
        print(f"New classification columns: {len(new_classification_cols)}")
        print(f"Total output columns: {len(classified_cols)}")
        
        if missing_original:
            print(f"\n❌ MISSING ORIGINAL COLUMNS:")
            for col in missing_original:
                print(f"  - {col}")
        else:
            print(f"\n✅ ALL ORIGINAL COLUMNS PRESERVED!")
        
        if new_classification_cols:
            print(f"\n✅ NEW CLASSIFICATION COLUMNS:")
            for col in sorted(new_classification_cols):
                print(f"  - {col}")
        
        # Test with CSV save/load
        test_csv_path = '/home/fiod/shimshi/test_verify_columns.csv'
        classified_df.to_csv(test_csv_path, index=False)
        
        # Check file size comparison
        original_size = len(original_df.to_csv(index=False))
        classified_size = len(classified_df.to_csv(index=False))
        
        print(f"\n=== SIZE COMPARISON ===")
        print(f"Original CSV size: {original_size:,} bytes")
        print(f"Classified CSV size: {classified_size:,} bytes")
        
        if classified_size > original_size:
            print(f"✅ SUCCESS: Classified CSV is larger (+{classified_size - original_size:,} bytes)")
        else:
            print(f"❌ ISSUE: Classified CSV is smaller (-{original_size - classified_size:,} bytes)")
        
        # Clean up
        if os.path.exists(test_csv_path):
            os.remove(test_csv_path)
        
        # Return success status
        success = len(missing_original) == 0 and classified_size > original_size
        return success
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    success = verify_column_preservation()
    
    print(f"\n{'='*50}")
    if success:
        print("🎉 SUCCESS: Column preservation fix is working!")
        print("All original columns are preserved and fraud CSV will be larger.")
    else:
        print("⚠️  ISSUE: Column preservation needs further investigation.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)