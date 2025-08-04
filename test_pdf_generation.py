#!/usr/bin/env python3
"""
Test script for PDF report generation module.

This script tests both the standalone PDF generator and the integrated pipeline.
"""

import os
import sys
from datetime import datetime

def test_standalone_pdf_generation():
    """Test standalone PDF generation."""
    print("Testing standalone PDF generation...")
    
    try:
        from pdf_report_generator import generate_pdf_report
        
        pdf_path = generate_pdf_report()
        
        if os.path.exists(pdf_path):
            file_size = os.path.getsize(pdf_path)
            print(f"✅ Standalone PDF generated successfully!")
            print(f"   File: {pdf_path}")
            print(f"   Size: {file_size:,} bytes")
            return True
        else:
            print(f"❌ PDF file not found: {pdf_path}")
            return False
            
    except Exception as e:
        print(f"❌ Standalone PDF generation failed: {e}")
        return False

def test_integrated_pdf_generation():
    """Test integrated PDF generation with pipeline."""
    print("\nTesting integrated PDF generation...")
    
    try:
        from main_pipeline import FraudDetectionPipeline
        import pandas as pd
        import json
        
        # Setup
        output_dir = '/home/fiod/shimshi/'
        data_path = '/home/fiod/shimshi/bq-results-20250804-141411-1754316868932.csv'
        
        # Load existing data
        quality_df = pd.read_csv('/home/fiod/shimshi/channel_quality_scores.csv')
        anomaly_df = pd.read_csv('/home/fiod/shimshi/channel_anomaly_scores.csv')
        
        with open('/home/fiod/shimshi/final_results.json', 'r') as f:
            final_results = json.load(f)
        
        # Test pipeline integration
        pipeline = FraudDetectionPipeline(data_path, output_dir)
        pdf_path = pipeline._generate_pdf_report(quality_df, final_results.get('cluster_summary', {}), anomaly_df)
        
        if pdf_path and not pdf_path.startswith("PDF generation failed"):
            if os.path.exists(pdf_path):
                file_size = os.path.getsize(pdf_path)
                print(f"✅ Integrated PDF generated successfully!")
                print(f"   File: {pdf_path}")
                print(f"   Size: {file_size:,} bytes")
                return True
            else:
                print(f"❌ PDF file not found: {pdf_path}")
                return False
        else:
            print(f"❌ Integrated PDF generation failed: {pdf_path}")
            return False
            
    except Exception as e:
        print(f"❌ Integrated PDF generation failed: {e}")
        return False

def check_requirements():
    """Check if all required files and data exist."""
    print("Checking requirements...")
    
    required_files = [
        '/home/fiod/shimshi/channel_quality_scores.csv',
        '/home/fiod/shimshi/channel_anomaly_scores.csv',
        '/home/fiod/shimshi/final_results.json',
        '/home/fiod/shimshi/pdf_report_generator.py',
        '/home/fiod/shimshi/main_pipeline.py'
    ]
    
    all_good = True
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {os.path.basename(file_path)}")
        else:
            print(f"❌ Missing: {file_path}")
            all_good = False
    
    return all_good

def main():
    """Run all tests."""
    print("=" * 60)
    print("PDF REPORT GENERATION TEST SUITE")
    print("=" * 60)
    print(f"Test run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Check requirements
    if not check_requirements():
        print("\n❌ Missing required files. Please run the main pipeline first.")
        return False
    
    print()
    
    # Test standalone generation
    standalone_success = test_standalone_pdf_generation()
    
    # Test integrated generation
    integrated_success = test_integrated_pdf_generation()
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)
    
    if standalone_success and integrated_success:
        print("✅ ALL TESTS PASSED!")
        print("   - Standalone PDF generation: ✅ WORKING")
        print("   - Integrated PDF generation: ✅ WORKING")
        print("   - Pipeline integration: ✅ WORKING")
        print()
        print("🎉 The PDF report generator is fully functional!")
        print("   You can now run the main pipeline to generate comprehensive PDF reports.")
        return True
    else:
        print("❌ SOME TESTS FAILED!")
        print(f"   - Standalone PDF generation: {'✅' if standalone_success else '❌'}")
        print(f"   - Integrated PDF generation: {'✅' if integrated_success else '❌'}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)