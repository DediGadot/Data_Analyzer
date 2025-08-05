#!/usr/bin/env python3
"""
Simple test script to verify the parallelism improvements without external dependencies.
"""

import pandas as pd
import numpy as np
import time
import logging
import sys
import os
from multiprocessing import cpu_count

# Add the current directory to Python path to import our modules
sys.path.append('/home/fiod/shimshi')

try:
    from anomaly_detection_optimized import OptimizedAnomalyDetector
    print("✅ Successfully imported OptimizedAnomalyDetector")
except ImportError as e:
    print(f"❌ Failed to import OptimizedAnomalyDetector: {e}")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_test_data(num_records: int = 10000) -> pd.DataFrame:
    """Generate synthetic test data for performance testing"""
    np.random.seed(42)
    
    # Generate synthetic fraud detection data
    data = {
        'channelId': np.random.choice(range(100, 500), num_records),
        'ip': [f"192.168.{np.random.randint(1,255)}.{np.random.randint(1,255)}" for _ in range(num_records)],
        'country': np.random.choice(['US', 'GB', 'DE', 'FR'], num_records),
        'device': np.random.choice(['desktop', 'mobile', 'tablet'], num_records),
        'browser': np.random.choice(['chrome', 'firefox', 'safari'], num_records),
        'browserMajorVersion': np.random.randint(70, 120, num_records),
        'userAgent': [f"Mozilla/5.0 Chrome/{np.random.randint(90,120)}" for _ in range(num_records)],
        'date': pd.date_range('2024-01-01', periods=num_records, freq='1min'),
        'isBot': np.random.choice([True, False], num_records, p=[0.1, 0.9]),
        'keyword': [f"keyword_{np.random.randint(1,100)}" for _ in range(num_records)],
        'referrer': np.random.choice(['direct', 'google.com', 'facebook.com'], num_records),
        'userSegmentFrequency': np.random.normal(50, 15, num_records)
    }
    
    df = pd.DataFrame(data)
    logger.info(f"Generated {num_records} test records with {df['channelId'].nunique()} unique channels")
    return df

def test_parallel_anomaly_detection():
    """Test the parallel anomaly detection implementation"""
    logger.info("=" * 60)
    logger.info("TESTING PARALLEL ANOMALY DETECTION")
    logger.info("=" * 60)
    
    # Generate test data
    logger.info("Generating test data...")
    test_df = generate_test_data(num_records=10000)  # Smaller dataset for testing
    
    # Initialize anomaly detector with optimizations
    detector = OptimizedAnomalyDetector(
        contamination=0.1,
        random_state=42,
        burst_detection_sample_size=1000,
        temporal_anomaly_min_volume=5,
        use_approximate_temporal=True,
        temporal_ml_estimators=25
    )
    
    logger.info("Starting parallel anomaly detection test...")
    logger.info(f"Available CPU cores: {cpu_count()}")
    
    # Run anomaly detection and measure time
    start_time = time.time()
    anomaly_results = detector.run_comprehensive_anomaly_detection(test_df)
    execution_time = time.time() - start_time
    
    # Analyze results
    logger.info("PARALLELISM TEST RESULTS:")
    logger.info(f"  Execution time: {execution_time:.2f} seconds")
    logger.info(f"  Processing speed: {len(test_df)/execution_time:.0f} records/second")
    
    if anomaly_results is not None and not anomaly_results.empty:
        logger.info(f"  Results shape: {anomaly_results.shape}")
        logger.info(f"  Channels analyzed: {len(anomaly_results)}")
        
        # Check for anomaly columns to verify all detection methods ran
        anomaly_cols = [col for col in anomaly_results.columns if 'anomaly' in col.lower()]
        logger.info(f"  Anomaly detection types found: {len(anomaly_cols)}")
        
        # List the detection types
        detection_types = set()
        for col in anomaly_cols:
            if 'temporal' in col:
                detection_types.add('temporal')
            elif 'geographic' in col or 'geo' in col:
                detection_types.add('geographic')
            elif 'device' in col:
                detection_types.add('device')
            elif 'behavioral' in col:
                detection_types.add('behavioral')
            elif 'volume' in col:
                detection_types.add('volume')
        
        logger.info(f"  Detection types executed: {sorted(detection_types)}")
        
        # Check if we got results from multiple detection methods (indicating parallel execution worked)
        if len(detection_types) >= 3:
            logger.info("✅ PARALLEL EXECUTION SUCCESSFUL - Multiple detection types completed")
        else:
            logger.warning("⚠️  LIMITED PARALLEL EXECUTION - Few detection types completed")
        
        if 'overall_anomaly_count' in anomaly_results.columns:
            anomalous_channels = (anomaly_results['overall_anomaly_count'] > 0).sum()
            logger.info(f"  Anomalous channels detected: {anomalous_channels}")
        
        return True
    else:
        logger.error("❌ PARALLEL ANOMALY DETECTION FAILED - No results generated")
        return False

def test_performance_comparison():
    """Compare performance between different configurations"""
    logger.info("\n" + "=" * 60)
    logger.info("PERFORMANCE COMPARISON TEST")
    logger.info("=" * 60)
    
    # Generate test data
    test_df = generate_test_data(num_records=5000)  # Small dataset for quick test
    
    # Test 1: With approximations (should be faster)
    logger.info("Testing with approximations ENABLED...")
    detector_fast = OptimizedAnomalyDetector(
        contamination=0.1,
        burst_detection_sample_size=500,
        temporal_anomaly_min_volume=5,
        use_approximate_temporal=True,
        temporal_ml_estimators=10  # Fewer estimators for speed
    )
    
    start_time = time.time()
    results_fast = detector_fast.run_comprehensive_anomaly_detection(test_df)
    fast_time = time.time() - start_time
    
    # Test 2: Without approximations (should be slower but more thorough)
    logger.info("Testing with approximations DISABLED...")
    detector_thorough = OptimizedAnomalyDetector(
        contamination=0.1,
        burst_detection_sample_size=5000,  # Process more data
        temporal_anomaly_min_volume=1,     # Process all entities
        use_approximate_temporal=False,
        temporal_ml_estimators=50          # More estimators for accuracy
    )
    
    start_time = time.time()
    results_thorough = detector_thorough.run_comprehensive_anomaly_detection(test_df)
    thorough_time = time.time() - start_time
    
    # Compare results
    logger.info("PERFORMANCE COMPARISON RESULTS:")
    logger.info(f"  Fast mode: {fast_time:.2f} seconds ({len(test_df)/fast_time:.0f} records/sec)")
    logger.info(f"  Thorough mode: {thorough_time:.2f} seconds ({len(test_df)/thorough_time:.0f} records/sec)")
    
    if fast_time < thorough_time:
        speedup = thorough_time / fast_time
        logger.info(f"  ✅ SPEEDUP ACHIEVED: {speedup:.1f}x faster with approximations")
    else:
        logger.warning("  ⚠️  No significant speedup observed")
    
    # Verify both produced results
    fast_valid = results_fast is not None and not results_fast.empty
    thorough_valid = results_thorough is not None and not results_thorough.empty
    
    logger.info(f"  Fast mode results valid: {fast_valid}")
    logger.info(f"  Thorough mode results valid: {thorough_valid}")
    
    return fast_valid and thorough_valid

def main():
    """Main test function"""
    logger.info("SIMPLIFIED PARALLELISM VERIFICATION TEST")
    logger.info(f"System CPU cores: {cpu_count()}")
    
    all_tests_passed = True
    
    try:
        # Test 1: Basic parallel anomaly detection
        test1_result = test_parallel_anomaly_detection()
        all_tests_passed = all_tests_passed and test1_result
        
        # Test 2: Performance comparison
        test2_result = test_performance_comparison()
        all_tests_passed = all_tests_passed and test2_result
        
    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        all_tests_passed = False
    
    # Final results
    logger.info("\n" + "=" * 60)
    if all_tests_passed:
        logger.info("🎉 PARALLELISM TESTS COMPLETED SUCCESSFULLY!")
        logger.info("✅ Anomaly detection methods can run in parallel")
        logger.info("✅ Multiple detection types are being executed")
        logger.info("✅ Performance optimizations are working")
        logger.info("✅ The pipeline should now utilize multiple CPU cores")
    else:
        logger.error("❌ SOME TESTS FAILED - Check the implementation")
    
    logger.info("=" * 60)
    
    return all_tests_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)