#!/usr/bin/env python3
"""
Test script for the updated PDF generator with Hebrew support
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

# Import the updated PDF generator
try:
    from pdf_report_generator_multilingual import MultilingualPDFReportGenerator
    logger.info("✓ Successfully imported MultilingualPDFReportGenerator")
except ImportError as e:
    logger.error(f"✗ Failed to import PDF generator: {e}")
    sys.exit(1)

def create_sample_data():
    """Create sample data for testing."""
    np.random.seed(42)
    
    # Create quality dataframe
    n_channels = 50
    quality_df = pd.DataFrame({
        'channelId': [f'channel_{i:03d}' for i in range(n_channels)],
        'quality_score': np.random.uniform(1, 10, n_channels),
        'bot_rate': np.random.uniform(0, 0.8, n_channels),
        'volume': np.random.lognormal(5, 2, n_channels).astype(int),
        'high_risk': np.random.choice([True, False], n_channels, p=[0.2, 0.8])
    })
    
    # Add quality categories
    quality_df['quality_category'] = pd.cut(
        quality_df['quality_score'], 
        bins=[0, 3, 5, 7, 10], 
        labels=['Low', 'Medium-Low', 'Medium-High', 'High']
    )
    
    # Create anomaly dataframe
    anomaly_df = pd.DataFrame({
        'channelId': quality_df['channelId'],
        'volume_anomaly': np.random.choice([True, False], n_channels, p=[0.3, 0.7]),
        'pattern_anomaly': np.random.choice([True, False], n_channels, p=[0.25, 0.75]),
        'time_anomaly': np.random.choice([True, False], n_channels, p=[0.2, 0.8]),
        'fraud_anomaly': np.random.choice([True, False], n_channels, p=[0.15, 0.85])
    })
    
    # Add overall anomaly count
    anomaly_cols = ['volume_anomaly', 'pattern_anomaly', 'time_anomaly', 'fraud_anomaly']
    anomaly_df['overall_anomaly_count'] = anomaly_df[anomaly_cols].sum(axis=1)
    anomaly_df['overall_anomaly_flag'] = anomaly_df['overall_anomaly_count'] > 0
    
    return quality_df, anomaly_df

def test_hebrew_pdf_generation():
    """Test Hebrew PDF generation with sample data."""
    logger.info("=== Testing Updated PDF Generator ===")
    
    try:
        # Create sample data
        logger.info("Creating sample data...")
        quality_df, anomaly_df = create_sample_data()
        
        # Initialize PDF generator
        logger.info("Initializing PDF generator...")
        pdf_generator = MultilingualPDFReportGenerator()
        
        # Create mock pipeline results
        pipeline_results = {
            'total_channels': len(quality_df),
            'high_risk_channels': len(quality_df[quality_df['high_risk'] == True]),
            'model_accuracy': 0.87,
            'processing_time': 45.2
        }
        
        final_results = {
            'summary': 'Test run completed successfully',
            'timestamp': datetime.now().isoformat()
        }
        
        # Generate reports
        logger.info("Generating PDF reports...")
        english_pdf, hebrew_pdf = pdf_generator.generate_comprehensive_report(
            quality_df=quality_df,
            anomaly_df=anomaly_df,
            final_results=final_results,
            pipeline_results=pipeline_results
        )
        
        # Check results
        results = {
            'english_pdf': english_pdf,
            'hebrew_pdf': hebrew_pdf,
            'english_exists': os.path.exists(english_pdf) if english_pdf else False,
            'hebrew_exists': os.path.exists(hebrew_pdf) if hebrew_pdf else False,
            'english_size': os.path.getsize(english_pdf) if english_pdf and os.path.exists(english_pdf) else 0,
            'hebrew_size': os.path.getsize(hebrew_pdf) if hebrew_pdf and os.path.exists(hebrew_pdf) else 0
        }
        
        return results
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main test function."""
    results = test_hebrew_pdf_generation()
    
    if results:
        logger.info("=== Test Results ===")
        logger.info(f"English PDF: {results['english_pdf']}")
        logger.info(f"Hebrew PDF: {results['hebrew_pdf']}")
        logger.info(f"English PDF exists: {'✓' if results['english_exists'] else '✗'}")
        logger.info(f"Hebrew PDF exists: {'✓' if results['hebrew_exists'] else '✗'}")
        
        if results['english_exists']:
            logger.info(f"English PDF size: {results['english_size']:,} bytes")
        if results['hebrew_exists']:
            logger.info(f"Hebrew PDF size: {results['hebrew_size']:,} bytes")
        
        # Success criteria
        success = (
            results['english_exists'] and results['hebrew_exists'] and
            results['english_size'] > 50000 and results['hebrew_size'] > 50000
        )
        
        if success:
            logger.info("✓ Test completed successfully!")
            logger.info("✓ Hebrew fonts are working properly in PDF generation")
            return 0
        else:
            logger.error("✗ Test failed - PDFs not generated properly")
            return 1
    else:
        logger.error("✗ Test failed completely")
        return 1

if __name__ == "__main__":
    sys.exit(main())