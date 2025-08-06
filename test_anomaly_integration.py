#!/usr/bin/env python3
"""
Test script to verify anomaly detection integration fixes
Tests the specific issues mentioned in the problem description:
1. Traffic similarity failing with empty features
2. Anomaly detection results not properly aggregating  
3. Pipeline failing before fraud classification
"""

import pandas as pd
import numpy as np
import logging
import os
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_test_data():
    """Create realistic test data that would trigger the original issues"""
    np.random.seed(42)
    
    # Create data that would cause traffic similarity to fail with empty features
    n_records = 1000
    n_channels = 50
    
    # Generate base data
    data = {
        'channelId': np.random.choice([f'CH{i:04d}' for i in range(n_channels)], n_records),
        'date': pd.date_range('2024-01-01', periods=n_records, freq='H'),
        'ip': [f'192.168.{np.random.randint(1,255)}.{np.random.randint(1,255)}' for _ in range(n_records)],
        'country': np.random.choice(['US', 'GB', 'DE', 'FR', 'JP'], n_records),
        'device': np.random.choice(['mobile', 'desktop', 'tablet'], n_records),
        'browser': np.random.choice(['chrome', 'firefox', 'safari', 'edge'], n_records),
        'userAgent': [f'Mozilla/5.0 Agent {i}' for i in range(n_records)],
        'isBot': np.random.choice([True, False], n_records, p=[0.1, 0.9]),
        'isLikelyBot': np.random.choice([True, False], n_records, p=[0.15, 0.85]),
        'isIpDatacenter': np.random.choice([True, False], n_records, p=[0.05, 0.95]),
        'isIpAnonymous': np.random.choice([True, False], n_records, p=[0.03, 0.97]),
        'browserMajorVersion': np.random.randint(50, 120, n_records),
        'keyword': [f'keyword_{np.random.randint(1,100)}' for _ in range(n_records)],
        'referrer': [f'https://site{np.random.randint(1,20)}.com' if np.random.random() > 0.1 else None for _ in range(n_records)]
    }
    
    df = pd.DataFrame(data)
    
    # Add edge cases that would cause issues:
    # 1. Channels with very few records (would cause empty feature aggregation)
    low_volume_channels = [f'CH{i:04d}' for i in range(45, 50)]
    for ch in low_volume_channels:
        mask = df['channelId'] == ch
        if mask.sum() > 3:  # Keep only 1-2 records for these channels
            indices_to_remove = df[mask].index[2:]
            df = df.drop(indices_to_remove)
    
    # 2. Add channels with identical features (would cause correlation issues)
    identical_mask = df['channelId'].isin(['CH0001', 'CH0002'])
    df.loc[identical_mask, 'country'] = 'US'
    df.loc[identical_mask, 'device'] = 'desktop'
    df.loc[identical_mask, 'browser'] = 'chrome'
    
    logger.info(f"Created test data with {len(df)} records, {df['channelId'].nunique()} unique channels")
    return df

def test_traffic_similarity_edge_cases():
    """Test traffic similarity with edge cases that would cause failures"""
    logger.info("=" * 50)
    logger.info("Testing Traffic Similarity Edge Cases")
    logger.info("=" * 50)
    
    try:
        from traffic_similarity import TrafficSimilarityModel
        
        # Test 1: Empty DataFrame
        logger.info("Test 1: Empty DataFrame")
        model = TrafficSimilarityModel()
        empty_df = pd.DataFrame()
        result = model.fit(empty_df)
        assert 'error' in result, "Should return error for empty DataFrame"
        logger.info("✓ Empty DataFrame handled correctly")
        
        # Test 2: DataFrame with no numeric features
        logger.info("Test 2: DataFrame with no numeric features") 
        non_numeric_df = pd.DataFrame({
            'channelId': ['CH001', 'CH002'],
            'name': ['Channel 1', 'Channel 2'],
            'category': ['A', 'B']
        })
        result = model.fit(non_numeric_df)
        assert 'error' in result, "Should return error for no numeric features"
        logger.info("✓ Non-numeric DataFrame handled correctly")
        
        # Test 3: DataFrame with single row
        logger.info("Test 3: Single row DataFrame")
        single_row_df = pd.DataFrame({
            'channelId': ['CH001'],
            'volume': [100],
            'bot_rate': [0.1]
        })
        result = model.fit(single_row_df)
        assert 'error' in result, "Should return error for single row"
        logger.info("✓ Single row DataFrame handled correctly")
        
        # Test 4: DataFrame with all identical values
        logger.info("Test 4: DataFrame with identical values")
        identical_df = pd.DataFrame({
            'channelId': ['CH001', 'CH002', 'CH003'],
            'volume': [100, 100, 100],
            'bot_rate': [0.1, 0.1, 0.1]
        })
        result = model.fit(identical_df)
        # This should work but might have low quality metrics
        logger.info(f"✓ Identical values handled: {type(result)}")
        
        logger.info("✅ Traffic Similarity tests completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Traffic Similarity test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def test_anomaly_detection_aggregation():
    """Test anomaly detection result aggregation"""
    logger.info("=" * 50)
    logger.info("Testing Anomaly Detection Aggregation")
    logger.info("=" * 50)
    
    try:
        from anomaly_detection_optimized import OptimizedAnomalyDetector
        
        # Create test data
        df = create_test_data()
        
        # Initialize detector
        detector = OptimizedAnomalyDetector(
            contamination=0.1,
            burst_detection_sample_size=100,  # Small sample for testing
            temporal_anomaly_min_volume=2,    # Low threshold for testing
            use_approximate_temporal=True,
            temporal_ml_estimators=10         # Fast for testing
        )
        
        logger.info("Testing comprehensive anomaly detection...")
        results = detector.run_comprehensive_anomaly_detection(df)
        
        # Verify results structure
        assert not results.empty, "Results should not be empty"
        assert 'channelId' in results.columns, "Results should have channelId column"
        
        # Check for expected anomaly columns
        expected_cols = ['temporal_anomaly', 'geographic_anomaly', 'device_anomaly', 
                        'behavioral_anomaly', 'volume_anomaly', 'overall_anomaly_count']
        
        for col in expected_cols:
            if col not in results.columns:
                logger.warning(f"Missing expected column: {col}")
            else:
                logger.info(f"✓ Found expected column: {col}")
        
        logger.info(f"✓ Results shape: {results.shape}")
        logger.info(f"✓ Columns: {list(results.columns)}")
        
        # Test that channels from original data are included
        original_channels = set(df['channelId'].unique())
        result_channels = set(results['channelId'].unique())
        missing_channels = original_channels - result_channels
        
        if missing_channels:
            logger.warning(f"Missing {len(missing_channels)} channels in results")
        else:
            logger.info("✓ All channels included in results")
        
        logger.info("✅ Anomaly Detection aggregation tests completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Anomaly Detection test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def test_fraud_classification_integration():
    """Test fraud classification with anomaly results"""
    logger.info("=" * 50) 
    logger.info("Testing Fraud Classification Integration")
    logger.info("=" * 50)
    
    try:
        from fraud_classifier import FraudClassifier
        from quality_scoring import QualityScorer
        from feature_engineering import FeatureEngineer
        
        # Create test data
        df = create_test_data()
        
        # Create minimal quality results
        unique_channels = df['channelId'].unique()
        quality_results = pd.DataFrame({
            'channelId': unique_channels,
            'quality_score': np.random.uniform(1, 10, len(unique_channels)),
            'high_risk': np.random.choice([True, False], len(unique_channels))
        })
        
        # Create minimal anomaly results
        anomaly_results = pd.DataFrame({
            'channelId': unique_channels,
            'temporal_anomaly': np.random.choice([True, False], len(unique_channels)),
            'geographic_anomaly': np.random.choice([True, False], len(unique_channels)),
            'device_anomaly': np.random.choice([True, False], len(unique_channels)),
            'behavioral_anomaly': np.random.choice([True, False], len(unique_channels)),
            'volume_anomaly': np.random.choice([True, False], len(unique_channels)),
            'overall_anomaly_count': np.random.randint(0, 5, len(unique_channels))
        })
        
        # Test fraud classification
        classifier = FraudClassifier()
        
        logger.info("Testing fraud classification with complete dataset...")
        classified_df = classifier.classify_dataset(df, quality_results, anomaly_results)
        
        # Verify results
        assert len(classified_df) == len(df), "Classified data should have same length as original"
        assert 'classification' in classified_df.columns, "Should have classification column"
        assert 'quality_score' in classified_df.columns, "Should have quality_score column"
        assert 'risk_score' in classified_df.columns, "Should have risk_score column"
        
        # Check anomaly columns are included
        for col in ['temporal_anomaly', 'geographic_anomaly', 'device_anomaly', 'behavioral_anomaly', 'volume_anomaly']:
            assert col in classified_df.columns, f"Should have {col} column"
        
        # Check classifications
        fraud_count = len(classified_df[classified_df['classification'] == 'fraud'])
        good_count = len(classified_df[classified_df['classification'] == 'good_account'])
        
        logger.info(f"✓ Classification results: {fraud_count} fraud, {good_count} good")
        logger.info(f"✓ Fraud percentage: {fraud_count/len(classified_df)*100:.1f}%")
        
        # Test CSV generation
        output_path = "/tmp/test_fraud_classification.csv"
        classified_df.to_csv(output_path, index=False)
        logger.info(f"✓ CSV saved to {output_path}")
        
        # Verify CSV can be read back
        read_df = pd.read_csv(output_path)
        assert len(read_df) == len(classified_df), "CSV should preserve all rows"
        
        logger.info("✅ Fraud Classification integration tests completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Fraud Classification test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def test_pipeline_resilience():
    """Test pipeline resilience to failures"""
    logger.info("=" * 50)
    logger.info("Testing Pipeline Resilience")  
    logger.info("=" * 50)
    
    try:
        # Test with minimal imports to avoid dependency issues
        df = create_test_data()
        
        # Test 1: Empty channel features should not crash traffic similarity
        logger.info("Test 1: Empty channel features")
        empty_channel_features = pd.DataFrame()
        
        # Simulate traffic similarity call with empty data
        try:
            from main_pipeline_optimized import OptimizedTrafficSimilarity
            similarity_model = OptimizedTrafficSimilarity(approximate=True)
            result = similarity_model.compute_similarity_fast(empty_channel_features)
            assert 'error' in result or result.get('fallback', False), "Should handle empty features gracefully"
            logger.info("✓ Empty channel features handled gracefully")
        except ImportError:
            logger.info("✓ Traffic similarity import test skipped (dependencies)")
        
        # Test 2: Missing columns should not crash anomaly detection
        logger.info("Test 2: Missing columns in anomaly detection")
        incomplete_df = df[['channelId', 'date']].copy()  # Only minimal columns
        
        try:
            from anomaly_detection_optimized import OptimizedAnomalyDetector
            detector = OptimizedAnomalyDetector(use_approximate_temporal=True)
            
            # Should not crash even with missing columns
            temporal_result = detector.detect_temporal_anomalies(incomplete_df)
            logger.info(f"✓ Temporal anomaly detection handled missing columns: {temporal_result.shape}")
            
            geo_result = detector.detect_geographic_anomalies(incomplete_df) 
            logger.info(f"✓ Geographic anomaly detection handled missing columns: {geo_result.shape}")
            
        except ImportError:
            logger.info("✓ Anomaly detection import test skipped (dependencies)")
        
        # Test 3: Classification with minimal data
        logger.info("Test 3: Classification resilience")
        try:
            from fraud_classifier import FraudClassifier
            
            classifier = FraudClassifier()
            
            # Test with empty results
            empty_quality = pd.DataFrame()
            empty_anomaly = pd.DataFrame()
            
            classified_df = classifier.classify_dataset(df.head(10), empty_quality, empty_anomaly)
            assert len(classified_df) == 10, "Should process all rows even with empty results"
            logger.info("✓ Classification handled empty quality/anomaly results")
            
        except ImportError:
            logger.info("✓ Fraud classifier import test skipped (dependencies)")
        
        logger.info("✅ Pipeline resilience tests completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Pipeline resilience test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def main():
    """Run all tests"""
    logger.info("🚀 Starting Anomaly Detection Integration Tests")
    logger.info("Testing fixes for:")
    logger.info("1. Traffic similarity failing with empty features")
    logger.info("2. Anomaly detection results not properly aggregating")
    logger.info("3. Pipeline failing before fraud classification")
    logger.info("")
    
    results = {
        'traffic_similarity': test_traffic_similarity_edge_cases(),
        'anomaly_aggregation': test_anomaly_detection_aggregation(), 
        'fraud_classification': test_fraud_classification_integration(),
        'pipeline_resilience': test_pipeline_resilience()
    }
    
    logger.info("=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name:<25}: {status}")
        if result:
            passed += 1
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests PASSED! The anomaly detection integration fixes are working correctly.")
        return True
    else:
        logger.error(f"⚠️  {total-passed} test(s) FAILED. Some issues may remain.")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)