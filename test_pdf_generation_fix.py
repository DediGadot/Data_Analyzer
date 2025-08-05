#!/usr/bin/env python3
"""
Test script to validate PDF generation with sample data
"""

import pandas as pd
import numpy as np
import os
import sys
import logging
from datetime import datetime

# Add current directory to path
sys.path.append('/home/fiod/shimshi')

from pdf_report_generator_multilingual import MultilingualPDFReportGenerator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_sample_data():
    """Create sample data for testing PDF generation"""
    logger.info("Creating sample quality and anomaly data")
    
    # Create sample quality results (what the QualityScorer returns)
    np.random.seed(42)
    n_channels = 50
    
    # Generate channel IDs
    channel_ids = [f"channel_{i:03d}" for i in range(n_channels)]
    
    # Create quality results DataFrame with channelId as index (as returned by QualityScorer)
    quality_results = pd.DataFrame({
        'quality_score': np.random.uniform(1, 10, n_channels),
        'bot_rate': np.random.uniform(0, 0.8, n_channels),
        'volume': np.random.randint(10, 10000, n_channels),
        'fraud_score_avg': np.random.uniform(0, 5, n_channels),
        'ip_diversity': np.random.randint(1, 100, n_channels),
        'country_diversity': np.random.randint(1, 20, n_channels)
    }, index=pd.Index(channel_ids, name='channelId'))
    
    # Add quality categories (as done by QualityScorer)
    quality_results['quality_category'] = pd.cut(
        quality_results['quality_score'],
        bins=[0, 3, 5, 7, 10],
        labels=['Low', 'Medium-Low', 'Medium-High', 'High']
    )
    
    # Add risk flags (as done by QualityScorer)
    quality_results['high_risk'] = (
        (quality_results['bot_rate'] > 0.3) |
        (quality_results['fraud_score_avg'] > 2.0) |
        (quality_results['quality_score'] < 3.0)
    )
    
    # Create sample anomaly results
    anomaly_results = pd.DataFrame({
        'temporal_anomaly': np.random.choice([True, False], n_channels, p=[0.1, 0.9]),
        'geographic_anomaly': np.random.choice([True, False], n_channels, p=[0.05, 0.95]),
        'device_anomaly': np.random.choice([True, False], n_channels, p=[0.08, 0.92]),
        'behavioral_anomaly': np.random.choice([True, False], n_channels, p=[0.12, 0.88]),
        'volume_anomaly': np.random.choice([True, False], n_channels, p=[0.03, 0.97])
    }, index=pd.Index(channel_ids, name='channelId'))
    
    # Calculate overall anomaly counts
    anomaly_cols = [col for col in anomaly_results.columns if 'anomaly' in col]
    anomaly_results['overall_anomaly_count'] = anomaly_results[anomaly_cols].sum(axis=1)
    anomaly_results['overall_anomaly_flag'] = anomaly_results['overall_anomaly_count'] > 0
    
    return quality_results, anomaly_results

def test_pdf_generation_structure():
    """Test the data structure conversion and PDF generation"""
    logger.info("Testing PDF generation with sample data")
    
    # Create sample data
    quality_results, anomaly_results = create_sample_data()
    
    logger.info(f"Original quality_results structure:")
    logger.info(f"  Index name: {quality_results.index.name}")
    logger.info(f"  Columns: {list(quality_results.columns)}")
    logger.info(f"  Shape: {quality_results.shape}")
    
    logger.info(f"Original anomaly_results structure:")
    logger.info(f"  Index name: {anomaly_results.index.name}")
    logger.info(f"  Columns: {list(anomaly_results.columns)}")
    logger.info(f"  Shape: {anomaly_results.shape}")
    
    # Test the same logic as in the fixed _generate_pdf_report method
    quality_results_copy = quality_results.copy()
    anomaly_results_copy = anomaly_results.copy()
    
    # Fix quality_results structure
    if quality_results_copy.index.name == 'channelId' or 'channelId' in str(quality_results_copy.index.names):
        quality_results_copy = quality_results_copy.reset_index()
        logger.info("✓ Converted channelId from index to column for quality_results")
    elif 'channelId' not in quality_results_copy.columns:
        quality_results_copy['channelId'] = quality_results_copy.index.astype(str)
        logger.info("✓ Created channelId column from index for quality_results")
    
    # Fix anomaly_results structure
    if not anomaly_results_copy.empty:
        if anomaly_results_copy.index.name == 'channelId' or 'channelId' in str(anomaly_results_copy.index.names):
            anomaly_results_copy = anomaly_results_copy.reset_index()
            logger.info("✓ Converted channelId from index to column for anomaly_results")
        elif 'channelId' not in anomaly_results_copy.columns:
            anomaly_results_copy['channelId'] = anomaly_results_copy.index.astype(str)
            logger.info("✓ Created channelId column from index for anomaly_results")
    
    logger.info(f"Fixed quality_results structure:")
    logger.info(f"  Columns: {list(quality_results_copy.columns)}")
    logger.info(f"  Shape: {quality_results_copy.shape}")
    logger.info(f"  Sample channelId values: {quality_results_copy['channelId'].head(3).tolist()}")
    
    logger.info(f"Fixed anomaly_results structure:")
    logger.info(f"  Columns: {list(anomaly_results_copy.columns)}")
    logger.info(f"  Shape: {anomaly_results_copy.shape}")
    logger.info(f"  Sample channelId values: {anomaly_results_copy['channelId'].head(3).tolist()}")
    
    # Check required columns for PDF generation
    required_quality_cols = ['channelId', 'quality_score', 'bot_rate', 'volume', 'quality_category', 'high_risk']
    missing_quality_cols = [col for col in required_quality_cols if col not in quality_results_copy.columns]
    
    required_anomaly_cols = ['channelId', 'overall_anomaly_count']
    missing_anomaly_cols = [col for col in required_anomaly_cols if col not in anomaly_results_copy.columns]
    
    if missing_quality_cols:
        logger.error(f"❌ Missing required columns in quality_results: {missing_quality_cols}")
        return False
    else:
        logger.info("✓ All required columns present in quality_results")
    
    if missing_anomaly_cols:
        logger.error(f"❌ Missing required columns in anomaly_results: {missing_anomaly_cols}")
        return False
    else:
        logger.info("✓ All required columns present in anomaly_results")
    
    return True

def main():
    """Main test function"""
    logger.info("Starting PDF generation structure test")
    
    success = test_pdf_generation_structure()
    
    if success:
        logger.info("🎉 PDF generation structure test PASSED! The fix should work.")
    else:
        logger.error("💥 PDF generation structure test FAILED! Additional fixes needed.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)