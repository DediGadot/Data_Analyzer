#!/usr/bin/env python3
"""
Test script for Hebrew font rendering in PDF reports.
Tests both ReportLab and matplotlib Hebrew font support.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime

# Add current directory to path
sys.path.append('/home/fiod/shimshi')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_test_data():
    """Create synthetic test data for PDF generation."""
    np.random.seed(42)
    
    # Create quality data
    n_channels = 50
    quality_data = {
        'channelId': [f'ch_{i:03d}' for i in range(n_channels)],
        'quality_score': np.random.uniform(1, 10, n_channels),
        'bot_rate': np.random.uniform(0, 0.8, n_channels),
        'volume': np.random.lognormal(5, 2, n_channels).astype(int),
        'high_risk': np.random.choice([True, False], n_channels, p=[0.2, 0.8])
    }
    
    # Add quality categories
    quality_df = pd.DataFrame(quality_data)
    quality_df['quality_category'] = pd.cut(
        quality_df['quality_score'], 
        bins=[0, 2.5, 5, 7.5, 10], 
        labels=['Low', 'Medium-Low', 'Medium-High', 'High']
    )
    
    # Create anomaly data
    anomaly_data = {
        'channelId': quality_df['channelId'].copy(),
        'volume_anomaly': np.random.choice([True, False], n_channels, p=[0.3, 0.7]),
        'rate_anomaly': np.random.choice([True, False], n_channels, p=[0.25, 0.75]),
        'pattern_anomaly': np.random.choice([True, False], n_channels, p=[0.2, 0.8]),
        'timing_anomaly': np.random.choice([True, False], n_channels, p=[0.15, 0.85])
    }
    
    anomaly_df = pd.DataFrame(anomaly_data)
    anomaly_df['overall_anomaly_count'] = (
        anomaly_df['volume_anomaly'].astype(int) +
        anomaly_df['rate_anomaly'].astype(int) +
        anomaly_df['pattern_anomaly'].astype(int) +
        anomaly_df['timing_anomaly'].astype(int)
    )
    anomaly_df['overall_anomaly_flag'] = anomaly_df['overall_anomaly_count'] > 0
    
    return quality_df, anomaly_df

def test_hebrew_font_registration():
    """Test Hebrew font registration."""
    try:
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        
        # Test font registration
        fonts_to_test = [
            '/home/fiod/shimshi/fonts/NotoSansHebrew.ttf',
            '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
        ]
        
        for font_path in fonts_to_test:
            if os.path.exists(font_path):
                try:
                    pdfmetrics.registerFont(TTFont('TestHebrewFont', font_path))
                    logger.info(f"✓ Successfully registered font: {font_path}")
                    return True
                except Exception as e:
                    logger.warning(f"✗ Failed to register font {font_path}: {e}")
        
        logger.error("✗ No Hebrew fonts could be registered")
        return False
        
    except ImportError as e:
        logger.error(f"✗ ReportLab not available: {e}")
        return False

def test_bidi_support():
    """Test RTL text processing."""
    try:
        from bidi.algorithm import get_display
        
        # Test Hebrew text processing
        hebrew_text = "שלום עולם - בדיקת RTL"
        processed_text = get_display(hebrew_text)
        
        logger.info(f"✓ RTL processing works")
        logger.info(f"  Original: {hebrew_text}")
        logger.info(f"  Processed: {processed_text}")
        return True
        
    except ImportError as e:
        logger.error(f"✗ python-bidi not available: {e}")
        return False

def test_matplotlib_hebrew():
    """Test matplotlib Hebrew text support."""
    try:
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        import matplotlib.pyplot as plt
        import matplotlib.font_manager as fm
        
        # Configure matplotlib for Hebrew
        plt.rcParams['font.family'] = ['DejaVu Sans', 'Noto Sans Hebrew', 'Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False
        
        # Test Hebrew text rendering
        fig, ax = plt.subplots(figsize=(8, 6))
        hebrew_text = "בדיקת טקסט עברי ב-matplotlib"
        english_text = "English text test"
        
        ax.text(0.5, 0.7, hebrew_text, ha='center', va='center', fontsize=14)
        ax.text(0.5, 0.3, english_text, ha='center', va='center', fontsize=14)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title("Hebrew Font Test")
        
        # Save test plot
        test_path = '/home/fiod/shimshi/test_hebrew_matplotlib.png'
        plt.savefig(test_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        if os.path.exists(test_path):
            logger.info(f"✓ Matplotlib Hebrew test plot created: {test_path}")
            return True
        else:
            logger.error("✗ Failed to create matplotlib test plot")
            return False
            
    except Exception as e:
        logger.error(f"✗ Matplotlib Hebrew test failed: {e}")
        return False

def test_pdf_generation():
    """Test complete PDF generation with Hebrew support."""
    try:
        from pdf_report_generator_multilingual import MultilingualPDFReportGenerator
        
        logger.info("Creating test data...")
        quality_df, anomaly_df = create_test_data()
        
        # Create pipeline results mock
        pipeline_results = {
            'model_performance': {
                'quality_model_accuracy': 0.85,
                'anomaly_detection_precision': 0.78,
                'processing_time_seconds': 45.2
            },
            'data_summary': {
                'total_channels_processed': len(quality_df),
                'date_range': '2025-07-01 to 2025-08-05'
            }
        }
        
        final_results = {
            'summary': {
                'total_channels': len(quality_df),
                'high_risk_channels': len(quality_df[quality_df['high_risk'] == True]),
                'anomalous_channels': len(anomaly_df[anomaly_df['overall_anomaly_count'] > 0])
            }
        }
        
        logger.info("Initializing PDF generator...")
        generator = MultilingualPDFReportGenerator('/home/fiod/shimshi/')
        
        # Test Hebrew font availability
        if generator.hebrew_font_available:
            logger.info("✓ Hebrew fonts available in PDF generator")
        else:
            logger.warning("⚠ Hebrew fonts not fully available, using fallback")
        
        logger.info("Generating test PDF reports...")
        english_path, hebrew_path = generator.generate_comprehensive_report(
            quality_df, anomaly_df, final_results, pipeline_results
        )
        
        # Check results
        success = True
        if english_path and os.path.exists(english_path):
            logger.info(f"✓ English PDF generated: {english_path}")
        else:
            logger.error("✗ English PDF generation failed")
            success = False
        
        if hebrew_path and os.path.exists(hebrew_path):
            logger.info(f"✓ Hebrew PDF generated: {hebrew_path}")
            # Check file size (should be reasonable)
            file_size = os.path.getsize(hebrew_path)
            if file_size > 100000:  # At least 100KB
                logger.info(f"✓ Hebrew PDF file size looks good: {file_size:,} bytes")
            else:
                logger.warning(f"⚠ Hebrew PDF file size may be too small: {file_size:,} bytes")
        else:
            logger.error("✗ Hebrew PDF generation failed")
            success = False
        
        return success, english_path, hebrew_path
        
    except Exception as e:
        logger.error(f"✗ PDF generation test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False, None, None

def test_hebrew_text_processing():
    """Test Hebrew text processing functions."""
    try:
        from pdf_report_generator_multilingual import MultilingualPDFReportGenerator
        
        generator = MultilingualPDFReportGenerator('/home/fiod/shimshi/')
        
        # Test various Hebrew texts
        test_texts = [
            "דוח צינור ML לזיהוי הונאות",
            "ניתוח מקיף ותובנות", 
            "סיכום מנהלים",
            "ממצאים עיקריים",
            "התפלגות איכות ערוצים",
            "שיעור בוטים ממוצע: 15.5%",
            "מספר ערוצים בסיכון גבוה: 25"
        ]
        
        logger.info("Testing Hebrew text processing...")
        for i, text in enumerate(test_texts):
            processed = generator._process_hebrew_text(text)
            translated = generator.t('title', 'he')  # Test translation function
            logger.info(f"  Test {i+1}: '{text}' -> '{processed}'")
        
        logger.info("✓ Hebrew text processing completed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Hebrew text processing test failed: {e}")
        return False

def main():
    """Run all Hebrew font tests."""
    logger.info("="*60)
    logger.info("Hebrew Font Rendering Test Suite")
    logger.info("="*60)
    
    results = {}
    
    # Test 1: Font registration
    logger.info("\n1. Testing Hebrew font registration...")
    results['font_registration'] = test_hebrew_font_registration()
    
    # Test 2: RTL text support
    logger.info("\n2. Testing RTL text processing...")
    results['bidi_support'] = test_bidi_support()
    
    # Test 3: Matplotlib Hebrew support
    logger.info("\n3. Testing matplotlib Hebrew support...")
    results['matplotlib_hebrew'] = test_matplotlib_hebrew()
    
    # Test 4: Hebrew text processing
    logger.info("\n4. Testing Hebrew text processing...")
    results['text_processing'] = test_hebrew_text_processing()
    
    # Test 5: Complete PDF generation
    logger.info("\n5. Testing complete PDF generation...")
    pdf_success, english_path, hebrew_path = test_pdf_generation()
    results['pdf_generation'] = pdf_success
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("TEST RESULTS SUMMARY")
    logger.info("="*60)
    
    for test_name, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        logger.info(f"{test_name.replace('_', ' ').title()}: {status}")
    
    total_tests = len(results)
    passed_tests = sum(results.values())
    
    logger.info(f"\nOverall: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        logger.info("🎉 All tests passed! Hebrew font rendering should work correctly.")
    elif passed_tests >= total_tests * 0.8:
        logger.info("⚠ Most tests passed. Hebrew rendering should work with minor issues.")
    else:
        logger.info("❌ Several tests failed. Hebrew rendering may have significant issues.")
    
    # Provide paths to generated files
    if pdf_success:
        logger.info(f"\nGenerated files:")
        if english_path:
            logger.info(f"  English PDF: {english_path}")
        if hebrew_path:
            logger.info(f"  Hebrew PDF: {hebrew_path}")
        if os.path.exists('/home/fiod/shimshi/test_hebrew_matplotlib.png'):
            logger.info(f"  Test plot: /home/fiod/shimshi/test_hebrew_matplotlib.png")
    
    return passed_tests == total_tests

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)