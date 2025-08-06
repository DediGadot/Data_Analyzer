#!/usr/bin/env python3
"""
Test script to verify the anomaly detection fixes work correctly
"""

import pandas as pd
import numpy as np
import sys
import os
sys.path.append('/home/fiod/shimshi')

from anomaly_detection_optimized import OptimizedAnomalyDetector

def create_test_data():
    """Create test data matching the actual CSV structure"""
    np.random.seed(42)
    
    # Create test data with the actual column structure
    n_rows = 100
    data = {
        'date': pd.date_range('2025-08-01', periods=n_rows, freq='H'),
        'keyword': np.random.choice(['keyword1', 'keyword2', 'keyword3', None], n_rows),
        'country': np.random.choice(['US', 'CA', 'UK'], n_rows),
        'browser': np.random.choice(['chrome', 'firefox', 'safari'], n_rows),
        'device': np.random.choice(['mobile', 'desktop', 'tablet'], n_rows),
        'referrer': np.random.choice(['google.com', 'facebook.com', None], n_rows),
        'ip': [f'192.168.{np.random.randint(1,255)}.{np.random.randint(1,255)}' for _ in range(n_rows)],
        'publisherId': [f'pub_{i%5}' for i in range(n_rows)],
        'channelId': [f'ch_{i%10}' for i in range(n_rows)],
        'advertiserId': [f'adv_{i%3}' for i in range(n_rows)],
        'feedId': [f'feed_{i%4}' for i in range(n_rows)],
        'browserMajorVersion': np.random.choice([90, 91, 92, 93, 94], n_rows),
        'userId': [f'user_{i%20}' for i in range(n_rows)],  # This is the correct column name
        'isLikelyBot': np.random.choice([True, False], n_rows),
        'ipClassification': np.random.choice(['clean', 'suspicious'], n_rows),
        'isIpDatacenter': np.random.choice([True, False], n_rows),
        'datacenterName': np.random.choice(['AWS', 'Google', None], n_rows),
        'ipHostName': [f'host_{i}.com' for i in range(n_rows)],
        'isIpAnonymous': np.random.choice([True, False], n_rows),
        'isIpCrawler': np.random.choice([True, False], n_rows),
        'isIpPublicProxy': np.random.choice([True, False], n_rows),
        'isIpVPN': np.random.choice([True, False], n_rows),
        'isIpHostingService': np.random.choice([True, False], n_rows),
        'isIpTOR': np.random.choice([True, False], n_rows),
        'isIpResidentialProxy': np.random.choice([True, False], n_rows),
        'performance': np.random.uniform(0.1, 2.0, n_rows),
        'detection': np.random.choice(['detected', 'undetected'], n_rows),
        'platform': np.random.choice(['web', 'mobile', 'api'], n_rows),
        'location': np.random.choice(['US-CA', 'US-NY', 'CA-ON'], n_rows),
        'userAgent': [f'Mozilla/5.0 Agent {i}' for i in range(n_rows)]
    }
    
    return pd.DataFrame(data)

def test_behavioral_anomaly_detection():
    """Test that behavioral anomaly detection works without .str accessor error"""
    print("=== Testing Behavioral Anomaly Detection ===")
    
    df = create_test_data()
    detector = OptimizedAnomalyDetector(use_approximate_temporal=True)
    
    try:
        # This should NOT fail with .str accessor error
        results = detector.detect_behavioral_anomalies(df)
        
        if results.empty:
            print("⚠️  No behavioral anomalies detected (expected for small test data)")
        else:
            print(f"✅ Behavioral anomaly detection completed successfully!")
            print(f"   Results shape: {results.shape}")
            print(f"   Columns: {list(results.columns)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Behavioral anomaly detection failed: {e}")
        return False

def test_volume_anomaly_detection():
    """Test that volume anomaly detection works without 'user' column error"""
    print("\n=== Testing Volume Anomaly Detection ===")
    
    df = create_test_data()
    detector = OptimizedAnomalyDetector(use_approximate_temporal=True)
    
    try:
        # This should NOT fail with "Column(s) ['user'] do not exist" error
        results = detector.detect_volume_anomalies(df)
        
        if results.empty:
            print("⚠️  No volume anomalies detected (expected for small test data)")
        else:
            print(f"✅ Volume anomaly detection completed successfully!")
            print(f"   Results shape: {results.shape}")
            print(f"   Columns: {list(results.columns)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Volume anomaly detection failed: {e}")
        return False

def test_comprehensive_anomaly_detection():
    """Test the full comprehensive anomaly detection"""
    print("\n=== Testing Comprehensive Anomaly Detection ===")
    
    df = create_test_data()
    detector = OptimizedAnomalyDetector(use_approximate_temporal=True)
    
    try:
        # This should run all anomaly detection methods
        results = detector.run_comprehensive_anomaly_detection(df)
        
        print(f"✅ Comprehensive anomaly detection completed!")
        print(f"   Results shape: {results.shape}")
        if not results.empty:
            print(f"   Columns: {list(results.columns)}")
            
            # Check for anomaly counts
            if 'overall_anomaly_count' in results.columns:
                total_anomalies = results['overall_anomaly_count'].sum()
                print(f"   Total anomalies detected: {total_anomalies}")
        
        return True
        
    except Exception as e:
        print(f"❌ Comprehensive anomaly detection failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("=== ANOMALY DETECTION FIXES TEST ===")
    print("Testing the fixes for:")
    print("1. Behavioral anomaly .str accessor error")
    print("2. Volume anomaly missing 'user' column error")
    print()
    
    # Run all tests
    tests = [
        test_behavioral_anomaly_detection,
        test_volume_anomaly_detection,
        test_comprehensive_anomaly_detection
    ]
    
    passed_tests = 0
    for test_func in tests:
        try:
            if test_func():
                passed_tests += 1
        except Exception as e:
            print(f"❌ Test {test_func.__name__} failed with exception: {e}")
    
    print(f"\n=== TEST SUMMARY ===")
    print(f"Tests passed: {passed_tests}/{len(tests)}")
    
    if passed_tests == len(tests):
        print("🎉 ALL TESTS PASSED! The anomaly detection fixes are working correctly.")
        print("The optimized pipeline should now run without these errors.")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
    
    return passed_tests == len(tests)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)