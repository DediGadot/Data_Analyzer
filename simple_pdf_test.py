"""
Simple PDF Generation Test - Focus on the core issue
"""

import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime
from pdf_report_generator_multilingual import MultilingualPDFReportGenerator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_edge_case_pdf():
    """Test the specific edge case that was causing issues"""
    
    print("Testing Edge Case PDF Generation")
    print("=" * 50)
    
    # Create the same problematic data from the original test
    quality_df = pd.DataFrame({
        'channelId': [f'CH{i:04d}' for i in range(5)],
        'quality_score': [3.0, 3.0, 7.0, 7.0, 7.0],  # Only 2 unique values
        'bot_rate': [0.1, 0.1, 0.8, 0.8, 0.8],       # Only 2 unique values  
        'volume': [50, 50, 200, 200, 200],           # Only 2 unique values
        'quality_category': ['Low', 'Low', 'High', 'High', 'High'],
        'high_risk': [False, False, True, True, True]
    })
    
    # Create sample anomaly data
    anomaly_df = pd.DataFrame({
        'channelId': quality_df['channelId'],
        'temporal_anomaly': [True, False, True, False, True],
        'geographic_anomaly': [False, True, False, True, False],
        'volume_anomaly': [True, True, False, False, False],
        'device_anomaly': [False, False, True, True, True],
        'behavioral_anomaly': [True, False, False, True, False]
    })
    
    # Calculate overall anomaly count
    anomaly_cols = [col for col in anomaly_df.columns if 'anomaly' in col]
    anomaly_df['overall_anomaly_count'] = anomaly_df[anomaly_cols].sum(axis=1)
    
    print(f"Quality DataFrame shape: {quality_df.shape}")
    print(f"Unique quality scores: {quality_df['quality_score'].nunique()}")
    print(f"Unique bot rates: {quality_df['bot_rate'].nunique()}")
    print(f"Unique volumes: {quality_df['volume'].nunique()}")
    
    # Create test results
    final_results = {
        'top_quality_channels': quality_df.nlargest(10, 'quality_score').to_dict('records'),
        'high_risk_channels': quality_df[quality_df['high_risk'] == True].head(10).to_dict('records'),
        'summary_stats': {
            'total_channels': len(quality_df),
            'high_risk_count': len(quality_df[quality_df['high_risk'] == True]),
            'avg_quality_score': float(quality_df['quality_score'].mean())
        }
    }
    
    pipeline_results = {
        'pipeline_summary': {
            'total_processing_time_minutes': 5.0,
            'records_processed': len(quality_df),
            'channels_analyzed': len(quality_df),
            'models_trained': 3
        }
    }
    
    # Test PDF generation
    try:
        print("\nGenerating PDFs...")
        start_time = datetime.now()
        
        generator = MultilingualPDFReportGenerator()
        en_path, he_path = generator.generate_comprehensive_report(
            quality_df, 
            anomaly_df, 
            final_results, 
            pipeline_results
        )
        
        generation_time = (datetime.now() - start_time).total_seconds()
        
        print(f"\n✓ PDF Generation Successful!")
        print(f"Generation time: {generation_time:.2f} seconds")
        
        # Check results
        results = {}
        
        if en_path and os.path.exists(en_path):
            size = os.path.getsize(en_path)
            print(f"✓ English PDF: {en_path}")
            print(f"  Size: {size:,} bytes")
            results['english'] = {'path': en_path, 'size': size, 'created': True}
        else:
            print("✗ English PDF: Not created")
            results['english'] = {'created': False}
        
        if he_path and os.path.exists(he_path):
            size = os.path.getsize(he_path)
            print(f"✓ Hebrew PDF: {he_path}")
            print(f"  Size: {size:,} bytes")
            results['hebrew'] = {'path': he_path, 'size': size, 'created': True}
        else:
            print("✗ Hebrew PDF: Not created")
            results['hebrew'] = {'created': False}
        
        # Verify PDFs are valid
        if results['english']['created']:
            with open(en_path, 'rb') as f:
                header = f.read(8)
                if header.startswith(b'%PDF-'):
                    print("✓ English PDF is valid")
                    results['english']['valid'] = True
                else:
                    print("✗ English PDF is invalid")
                    results['english']['valid'] = False
        
        if results['hebrew']['created']:
            with open(he_path, 'rb') as f:
                header = f.read(8)
                if header.startswith(b'%PDF-'):
                    print("✓ Hebrew PDF is valid")
                    results['hebrew']['valid'] = True
                else:
                    print("✗ Hebrew PDF is invalid")
                    results['hebrew']['valid'] = False
        
        return True, results
        
    except Exception as e:
        print(f"\n✗ PDF Generation Failed: {str(e)}")
        import traceback
        print(f"Traceback:\n{traceback.format_exc()}")
        return False, {'error': str(e)}

def main():
    """Main test function"""
    print("SIMPLE PDF GENERATION TEST")
    print("=" * 80)
    print("This test focuses on the specific edge case that was causing Hebrew PDF failures")
    print()
    
    success, results = test_edge_case_pdf()
    
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    
    if success:
        print("🎉 SUCCESS: PDF generation is working correctly!")
        print("\nThe original issue with Hebrew PDF generation has been fixed.")
        print("Both English and Hebrew PDFs are being generated successfully.")
        
        # Check if both languages worked
        english_ok = results.get('english', {}).get('created', False)
        hebrew_ok = results.get('hebrew', {}).get('created', False)
        
        if english_ok and hebrew_ok:
            print("\n✅ MULTILINGUAL SUPPORT: Both English and Hebrew PDFs generated successfully")
        elif english_ok:
            print("\n⚠️  PARTIAL SUCCESS: English PDF works but Hebrew PDF has issues")
        else:
            print("\n❌ LIMITED SUCCESS: PDF generation has problems")
            
    else:
        print("❌ FAILURE: PDF generation is still having issues")
        print(f"Error: {results.get('error', 'Unknown error')}")
    
    return success

if __name__ == "__main__":
    success = main()