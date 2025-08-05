#!/usr/bin/env python3
"""
Quick Hebrew font test for PDF generation.
"""

import os
import pandas as pd
import numpy as np
import sys
sys.path.append('/home/fiod/shimshi')

def test_hebrew_generation():
    """Test Hebrew PDF generation quickly."""
    try:
        from pdf_report_generator_multilingual import MultilingualPDFReportGenerator
        
        # Create minimal test data
        np.random.seed(42)
        quality_df = pd.DataFrame({
            'channelId': ['ch_001', 'ch_002', 'ch_003'],
            'quality_score': [8.5, 3.2, 6.7],
            'bot_rate': [0.1, 0.6, 0.3],
            'volume': [1000, 500, 2000],
            'high_risk': [False, True, False],
            'quality_category': ['High', 'Low', 'Medium-High']
        })
        
        anomaly_df = pd.DataFrame({
            'channelId': ['ch_001', 'ch_002', 'ch_003'],
            'volume_anomaly': [False, True, False],
            'rate_anomaly': [False, True, True],
            'overall_anomaly_count': [0, 2, 1],
            'overall_anomaly_flag': [False, True, True]
        })
        
        pipeline_results = {'model_performance': {'accuracy': 0.85}}
        final_results = {'summary': {'total_channels': 3}}
        
        print("Initializing PDF generator...")
        generator = MultilingualPDFReportGenerator('/home/fiod/shimshi/')
        
        print(f"Hebrew font available: {generator.hebrew_font_available}")
        
        print("Generating reports...")
        english_path, hebrew_path = generator.generate_comprehensive_report(
            quality_df, anomaly_df, final_results, pipeline_results
        )
        
        # Check results
        english_ok = english_path and os.path.exists(english_path)
        hebrew_ok = hebrew_path and os.path.exists(hebrew_path)
        
        print(f"\nResults:")
        print(f"English PDF: {'✓' if english_ok else '✗'} {english_path if english_path else 'Failed'}")
        print(f"Hebrew PDF: {'✓' if hebrew_ok else '✗'} {hebrew_path if hebrew_path else 'Failed'}")
        
        if hebrew_ok:
            file_size = os.path.getsize(hebrew_path)
            print(f"Hebrew PDF size: {file_size:,} bytes")
            
        return english_ok and hebrew_ok
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_hebrew_generation()
    print(f"\nTest {'PASSED' if success else 'FAILED'}")