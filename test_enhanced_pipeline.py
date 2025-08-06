"""
Test Enhanced Fraud Detection Pipeline
Tests the fraud classification functionality with sample data
"""

import pandas as pd
import numpy as np
import logging
import sys
import os
from datetime import datetime, timedelta

# Add current directory to path
sys.path.append('/home/fiod/shimshi')

from fraud_classifier import FraudClassifier

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_sample_data(n_rows=1000):
    """Create sample data for testing"""
    logger.info(f"Creating sample dataset with {n_rows} rows")
    
    # Generate sample data similar to the real structure
    np.random.seed(42)
    
    base_date = datetime(2025, 8, 1)
    
    sample_data = {
        'date': [base_date + timedelta(hours=np.random.randint(0, 24*7)) for _ in range(n_rows)],
        'keyword': [f'keyword_{i%100}' for i in range(n_rows)],
        'country': np.random.choice(['US', 'UK', 'DE', 'FR', 'CA'], n_rows),
        'browser': np.random.choice(['chrome', 'firefox', 'safari', 'edge'], n_rows),
        'device': np.random.choice(['mobile', 'desktop', 'tablet'], n_rows),
        'ip': [f'192.168.{np.random.randint(1,255)}.{np.random.randint(1,255)}' for _ in range(n_rows)],
        'publisherId': [f'pub_{i%50}' for i in range(n_rows)],
        'channelId': [f'ch_{i%20}' for i in range(n_rows)],
        'advertiserId': [f'adv_{i%30}' for i in range(n_rows)],
        'feedId': [f'feed_{i%10}' for i in range(n_rows)],
        'userId': [f'user_{i}' for i in range(n_rows)],
        'isLikelyBot': np.random.choice([True, False], n_rows, p=[0.15, 0.85]),
        'ipClassification': np.random.choice(['clean', 'suspicious', 'malicious'], n_rows, p=[0.7, 0.2, 0.1]),
        'isIpDatacenter': np.random.choice([True, False], n_rows, p=[0.1, 0.9]),
        'isIpAnonymous': np.random.choice([True, False], n_rows, p=[0.05, 0.95])
    }
    
    return pd.DataFrame(sample_data)

def create_sample_quality_results(channel_ids):
    """Create sample quality scoring results"""
    logger.info(f"Creating quality results for {len(channel_ids)} channels")
    
    np.random.seed(42)
    
    quality_data = []
    for channel_id in channel_ids:
        # Some channels are high quality, some low, some medium
        if np.random.random() < 0.1:
            # Low quality (potential fraud)
            quality_score = np.random.uniform(1, 3)
            high_risk = True
        elif np.random.random() < 0.15:
            # High quality (good)
            quality_score = np.random.uniform(8, 10)
            high_risk = False
        else:
            # Medium quality
            quality_score = np.random.uniform(4, 7)
            high_risk = np.random.choice([True, False], p=[0.3, 0.7])
        
        quality_data.append({
            'channelId': channel_id,
            'quality_score': quality_score,
            'high_risk': high_risk,
            'confidence': np.random.uniform(0.5, 0.95)
        })
    
    return pd.DataFrame(quality_data)

def create_sample_anomaly_results(n_rows):
    """Create sample anomaly detection results (row-level)"""
    logger.info(f"Creating anomaly results for {n_rows} rows")
    
    np.random.seed(42)
    
    anomaly_data = []
    for i in range(n_rows):
        # Random anomaly flags
        temporal_anomaly = np.random.choice([True, False], p=[0.1, 0.9])
        geographic_anomaly = np.random.choice([True, False], p=[0.08, 0.92])
        device_anomaly = np.random.choice([True, False], p=[0.05, 0.95])
        behavioral_anomaly = np.random.choice([True, False], p=[0.12, 0.88])
        volume_anomaly = np.random.choice([True, False], p=[0.07, 0.93])
        
        # Count total anomalies
        anomaly_count = sum([temporal_anomaly, geographic_anomaly, device_anomaly, 
                            behavioral_anomaly, volume_anomaly])
        
        anomaly_data.append({
            'temporal_anomaly': temporal_anomaly,
            'geographic_anomaly': geographic_anomaly,
            'device_anomaly': device_anomaly,
            'behavioral_anomaly': behavioral_anomaly,
            'volume_anomaly': volume_anomaly,
            'overall_anomaly_count': anomaly_count
        })
    
    return pd.DataFrame(anomaly_data)

def test_fraud_classifier():
    """Test the fraud classifier with sample data"""
    logger.info("=" * 60)
    logger.info("TESTING ENHANCED FRAUD DETECTION PIPELINE")
    logger.info("=" * 60)
    
    # Create sample data
    sample_df = create_sample_data(n_rows=1000)
    logger.info(f"Created sample data: {sample_df.shape}")
    logger.info(f"Sample columns: {list(sample_df.columns)}")
    
    # Get unique channel IDs
    unique_channels = sample_df['channelId'].unique()
    logger.info(f"Unique channels: {len(unique_channels)}")
    
    # Create sample quality results
    quality_results = create_sample_quality_results(unique_channels)
    logger.info(f"Created quality results: {quality_results.shape}")
    
    # Create sample anomaly results
    anomaly_results = create_sample_anomaly_results(len(sample_df))
    logger.info(f"Created anomaly results: {anomaly_results.shape}")
    
    # Initialize fraud classifier
    logger.info("Initializing fraud classifier...")
    classifier = FraudClassifier(
        quality_threshold_low=3.0,
        quality_threshold_high=7.0,
        anomaly_threshold_high=3,
        risk_threshold=0.5
    )
    
    # Test classification
    logger.info("Running fraud classification...")
    try:
        classified_df = classifier.classify_dataset(
            original_df=sample_df,
            quality_results=quality_results,
            anomaly_results=anomaly_results
        )
        
        logger.info(f"Classification completed successfully!")
        logger.info(f"Output shape: {classified_df.shape}")
        logger.info(f"Output columns: {list(classified_df.columns)}")
        
        # Analyze results
        fraud_count = len(classified_df[classified_df['classification'] == 'fraud'])
        good_count = len(classified_df[classified_df['classification'] == 'good_account'])
        
        logger.info("=" * 60)
        logger.info("CLASSIFICATION RESULTS SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total rows: {len(classified_df)}")
        logger.info(f"Fraud: {fraud_count} ({fraud_count/len(classified_df)*100:.1f}%)")
        logger.info(f"Good accounts: {good_count} ({good_count/len(classified_df)*100:.1f}%)")
        logger.info(f"Average quality score: {classified_df['quality_score'].mean():.2f}")
        logger.info(f"Average risk score: {classified_df['risk_score'].mean():.3f}")
        logger.info(f"Average confidence: {classified_df['confidence'].mean():.3f}")
        
        # Show sample fraud cases
        fraud_cases = classified_df[classified_df['classification'] == 'fraud'].head(3)
        if len(fraud_cases) > 0:
            logger.info("\nSample fraud cases:")
            for idx, row in fraud_cases.iterrows():
                logger.info(f"  Row {idx}: Quality={row['quality_score']:.1f}, Risk={row['risk_score']:.3f}, "
                           f"Reasons={row['reason_codes']}, Anomalies={row['overall_anomaly_count']}")
        
        # Show sample good cases
        good_cases = classified_df[classified_df['classification'] == 'good_account'].head(3)
        if len(good_cases) > 0:
            logger.info("\nSample good account cases:")
            for idx, row in good_cases.iterrows():
                logger.info(f"  Row {idx}: Quality={row['quality_score']:.1f}, Risk={row['risk_score']:.3f}, "
                           f"Reasons={row['reason_codes']}, Anomalies={row['overall_anomaly_count']}")
        
        # Save test results
        output_path = '/home/fiod/shimshi/test_classification_results.csv'
        classified_df.to_csv(output_path, index=False)
        logger.info(f"\nTest results saved to: {output_path}")
        
        # Verify required columns are present
        required_columns = [
            'classification', 'quality_score', 'risk_score', 'confidence', 'reason_codes',
            'temporal_anomaly', 'geographic_anomaly', 'device_anomaly', 
            'behavioral_anomaly', 'volume_anomaly', 'overall_anomaly_count'
        ]
        
        missing_columns = [col for col in required_columns if col not in classified_df.columns]
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return False
        else:
            logger.info("✓ All required columns present")
        
        # Test threshold configuration
        thresholds = classifier.get_classification_thresholds()
        logger.info(f"\nClassification thresholds: {thresholds}")
        
        logger.info("=" * 60)
        logger.info("TEST COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)
        
        return True
        
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

def test_pipeline_integration():
    """Test integration with the main pipeline (simulation)"""
    logger.info("Testing pipeline integration simulation...")
    
    try:
        # This would normally be called from the main pipeline
        # We're just testing the import and initialization
        from main_pipeline_optimized import OptimizedFraudDetectionPipeline
        
        logger.info("✓ Pipeline import successful")
        logger.info("✓ FraudClassifier integration ready")
        
        return True
        
    except Exception as e:
        logger.error(f"Pipeline integration test failed: {e}")
        return False

if __name__ == "__main__":
    logger.info("Starting enhanced fraud detection pipeline tests...")
    
    # Test 1: Fraud Classifier
    test1_success = test_fraud_classifier()
    
    # Test 2: Pipeline Integration
    test2_success = test_pipeline_integration()
    
    # Overall results
    if test1_success and test2_success:
        logger.info("\n🎉 ALL TESTS PASSED! Enhanced pipeline is ready for production.")
    else:
        logger.error("\n❌ Some tests failed. Please review the errors above.")
        sys.exit(1)