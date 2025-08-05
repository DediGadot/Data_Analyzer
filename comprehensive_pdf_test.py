"""
Comprehensive PDF Generation Test Script
Tests various edge cases and scenarios to ensure robust PDF generation
"""

import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime
from pdf_report_generator_multilingual import MultilingualPDFReportGenerator
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_test_case(name: str, quality_df: pd.DataFrame, anomaly_df: pd.DataFrame) -> dict:
    """Create a test case dictionary"""
    return {
        'name': name,
        'quality_df': quality_df,
        'anomaly_df': anomaly_df,
        'description': f"Test case: {name}"
    }

def generate_test_cases():
    """Generate comprehensive test cases"""
    test_cases = []
    
    # Test Case 1: Minimal data (edge case that was failing)
    print("Creating Test Case 1: Minimal Edge Case Data")
    quality_df_1 = pd.DataFrame({
        'channelId': ['CH0001', 'CH0002'],
        'quality_score': [3.0, 7.0],
        'bot_rate': [0.1, 0.8],
        'volume': [50, 200],
        'quality_category': ['Low', 'High'],
        'high_risk': [False, True]
    })
    
    anomaly_df_1 = pd.DataFrame({
        'channelId': ['CH0001', 'CH0002'],
        'temporal_anomaly': [True, False],
        'geographic_anomaly': [False, True],
        'volume_anomaly': [True, False],
        'device_anomaly': [False, True],
        'behavioral_anomaly': [True, False]
    })
    anomaly_df_1['overall_anomaly_count'] = anomaly_df_1.iloc[:, 1:].sum(axis=1)
    
    test_cases.append(create_test_case("Minimal Edge Case", quality_df_1, anomaly_df_1))
    
    # Test Case 2: Single value data
    print("Creating Test Case 2: Single Value Data")
    quality_df_2 = pd.DataFrame({
        'channelId': ['CH0001'],
        'quality_score': [5.0],
        'bot_rate': [0.5],
        'volume': [100],
        'quality_category': ['Medium'],
        'high_risk': [False]
    })
    
    anomaly_df_2 = pd.DataFrame({
        'channelId': ['CH0001'],
        'temporal_anomaly': [False],
        'geographic_anomaly': [False],
        'volume_anomaly': [False],
        'device_anomaly': [False],
        'behavioral_anomaly': [False]
    })
    anomaly_df_2['overall_anomaly_count'] = anomaly_df_2.iloc[:, 1:].sum(axis=1)
    
    test_cases.append(create_test_case("Single Value", quality_df_2, anomaly_df_2))
    
    # Test Case 3: All same values
    print("Creating Test Case 3: All Same Values")
    quality_df_3 = pd.DataFrame({
        'channelId': [f'CH{i:04d}' for i in range(10)],
        'quality_score': [5.0] * 10,
        'bot_rate': [0.5] * 10,
        'volume': [100] * 10,
        'quality_category': ['Medium'] * 10,
        'high_risk': [False] * 10
    })
    
    anomaly_df_3 = pd.DataFrame({
        'channelId': quality_df_3['channelId'],
        'temporal_anomaly': [False] * 10,
        'geographic_anomaly': [False] * 10,
        'volume_anomaly': [False] * 10,
        'device_anomaly': [False] * 10,
        'behavioral_anomaly': [False] * 10
    })
    anomaly_df_3['overall_anomaly_count'] = anomaly_df_3.iloc[:, 1:].sum(axis=1)
    
    test_cases.append(create_test_case("All Same Values", quality_df_3, anomaly_df_3))
    
    # Test Case 4: Extreme values
    print("Creating Test Case 4: Extreme Values")
    quality_df_4 = pd.DataFrame({
        'channelId': [f'CH{i:04d}' for i in range(5)],
        'quality_score': [0.0, 10.0, 0.0, 10.0, 5.0],
        'bot_rate': [0.0, 1.0, 0.0, 1.0, 0.5],
        'volume': [1, 1000000, 1, 1000000, 1000],
        'quality_category': ['Low', 'High', 'Low', 'High', 'Medium'],
        'high_risk': [True, False, True, False, False]
    })
    
    anomaly_df_4 = pd.DataFrame({
        'channelId': quality_df_4['channelId'],
        'temporal_anomaly': [True, False, True, False, True],
        'geographic_anomaly': [False, True, False, True, False],
        'volume_anomaly': [True, True, False, False, False],
        'device_anomaly': [False, False, True, True, True],
        'behavioral_anomaly': [True, False, False, True, False]
    })
    anomaly_df_4['overall_anomaly_count'] = anomaly_df_4.iloc[:, 1:].sum(axis=1)
    
    test_cases.append(create_test_case("Extreme Values", quality_df_4, anomaly_df_4))
    
    # Test Case 5: Normal distribution
    print("Creating Test Case 5: Normal Distribution")
    np.random.seed(42)
    n_channels = 100
    
    quality_df_5 = pd.DataFrame({
        'channelId': [f'CH{i:04d}' for i in range(n_channels)],
        'quality_score': np.random.normal(5, 2, n_channels).clip(0, 10),
        'bot_rate': np.random.beta(2, 5, n_channels),
        'volume': np.random.exponential(100, n_channels).astype(int),
        'quality_category': np.random.choice(['Low', 'Medium-Low', 'Medium-High', 'High'], n_channels),
        'high_risk': np.random.choice([True, False], n_channels, p=[0.2, 0.8])
    })
    
    anomaly_df_5 = pd.DataFrame({
        'channelId': quality_df_5['channelId'],
        'temporal_anomaly': np.random.choice([True, False], n_channels, p=[0.1, 0.9]),
        'geographic_anomaly': np.random.choice([True, False], n_channels, p=[0.1, 0.9]),
        'volume_anomaly': np.random.choice([True, False], n_channels, p=[0.1, 0.9]),
        'device_anomaly': np.random.choice([True, False], n_channels, p=[0.1, 0.9]),
        'behavioral_anomaly': np.random.choice([True, False], n_channels, p=[0.1, 0.9])
    })
    anomaly_df_5['overall_anomaly_count'] = anomaly_df_5.iloc[:, 1:].sum(axis=1)
    
    test_cases.append(create_test_case("Normal Distribution", quality_df_5, anomaly_df_5))
    
    # Test Case 6: With NaN values (error recovery test)
    print("Creating Test Case 6: NaN Values")
    quality_df_6 = pd.DataFrame({
        'channelId': ['CH0001', 'CH0002', 'CH0003'],
        'quality_score': [5.0, np.nan, 3.0],
        'bot_rate': [0.5, 0.7, np.nan],
        'volume': [100, 200, 300],
        'quality_category': ['Medium', None, 'Low'],
        'high_risk': [False, True, False]
    })
    
    anomaly_df_6 = pd.DataFrame({
        'channelId': ['CH0001', 'CH0002', 'CH0003'],
        'temporal_anomaly': [True, False, np.nan],
        'geographic_anomaly': [False, True, False],
        'volume_anomaly': [True, np.nan, False],
        'device_anomaly': [False, True, True],
        'behavioral_anomaly': [True, False, False]
    })
    # Handle NaN values in overall_anomaly_count calculation
    anomaly_df_6['overall_anomaly_count'] = anomaly_df_6.iloc[:, 1:].fillna(False).sum(axis=1)
    
    test_cases.append(create_test_case("NaN Values", quality_df_6, anomaly_df_6))
    
    return test_cases

def run_pdf_test(test_case: dict, output_dir: str = "/home/fiod/shimshi/") -> dict:
    """Run PDF generation test for a single test case"""
    result = {
        'name': test_case['name'],
        'success': False,
        'english_pdf': None,
        'hebrew_pdf': None,
        'error': None,
        'file_sizes': {},
        'generation_time': 0
    }
    
    try:
        logger.info(f"Testing: {test_case['name']}")
        
        # Create test results
        final_results = {
            'top_quality_channels': test_case['quality_df'].nlargest(min(10, len(test_case['quality_df'])), 'quality_score').to_dict('records'),
            'high_risk_channels': test_case['quality_df'][test_case['quality_df']['high_risk'] == True].head(10).to_dict('records'),
            'summary_stats': {
                'total_channels': len(test_case['quality_df']),
                'high_risk_count': len(test_case['quality_df'][test_case['quality_df']['high_risk'] == True]),
                'avg_quality_score': float(test_case['quality_df']['quality_score'].mean()) if not test_case['quality_df']['quality_score'].isna().all() else 0.0
            }
        }
        
        pipeline_results = {
            'pipeline_summary': {
                'total_processing_time_minutes': 2.0,
                'records_processed': len(test_case['quality_df']),
                'channels_analyzed': len(test_case['quality_df']),
                'models_trained': 3
            }
        }
        
        # Generate PDFs
        start_time = datetime.now()
        generator = MultilingualPDFReportGenerator(output_dir=output_dir)
        
        en_path, he_path = generator.generate_comprehensive_report(
            test_case['quality_df'], 
            test_case['anomaly_df'], 
            final_results, 
            pipeline_results
        )
        
        generation_time = (datetime.now() - start_time).total_seconds()
        
        result['english_pdf'] = en_path
        result['hebrew_pdf'] = he_path
        result['generation_time'] = generation_time
        
        # Check file sizes
        if en_path and os.path.exists(en_path):
            result['file_sizes']['english'] = os.path.getsize(en_path)
            
        if he_path and os.path.exists(he_path):
            result['file_sizes']['hebrew'] = os.path.getsize(he_path)
        
        result['success'] = True
        logger.info(f"✓ {test_case['name']}: SUCCESS")
        
    except Exception as e:
        result['error'] = str(e)
        result['traceback'] = traceback.format_exc()
        logger.error(f"✗ {test_case['name']}: FAILED - {str(e)}")
    
    return result

def main():
    """Run comprehensive PDF generation tests"""
    print("=" * 80)
    print("COMPREHENSIVE PDF GENERATION TEST SUITE")
    print("=" * 80)
    
    # Generate test cases
    test_cases = generate_test_cases()
    print(f"Generated {len(test_cases)} test cases")
    
    # Run tests
    results = []
    successful_tests = 0
    
    for test_case in test_cases:
        result = run_pdf_test(test_case)
        results.append(result)
        
        if result['success']:
            successful_tests += 1
    
    # Generate summary report
    print("\n" + "=" * 80)
    print("TEST RESULTS SUMMARY")
    print("=" * 80)
    
    print(f"Total tests: {len(test_cases)}")
    print(f"Successful: {successful_tests}")
    print(f"Failed: {len(test_cases) - successful_tests}")
    print(f"Success rate: {(successful_tests / len(test_cases)) * 100:.1f}%")
    
    print("\nDetailed Results:")
    print("-" * 40)
    
    for result in results:
        status = "✓ PASS" if result['success'] else "✗ FAIL"
        print(f"{status} | {result['name']:<20}")
        
        if result['success']:
            english_size = result['file_sizes'].get('english', 0)
            hebrew_size = result['file_sizes'].get('hebrew', 0)
            print(f"     | English PDF: {english_size:,} bytes")
            print(f"     | Hebrew PDF:  {hebrew_size:,} bytes")
            print(f"     | Generation time: {result['generation_time']:.2f}s")
        else:
            print(f"     | Error: {result['error']}")
        print()
    
    # Save detailed report
    report_file = "/home/fiod/shimshi/pdf_test_results.txt"
    with open(report_file, 'w') as f:
        f.write("COMPREHENSIVE PDF GENERATION TEST RESULTS\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated: {datetime.now()}\n\n")
        
        f.write(f"SUMMARY: {successful_tests}/{len(test_cases)} tests passed\n")
        f.write(f"Success rate: {(successful_tests / len(test_cases)) * 100:.1f}%\n\n")
        
        for result in results:
            f.write(f"Test: {result['name']}\n")
            f.write(f"Status: {'PASSED' if result['success'] else 'FAILED'}\n")
            
            if result['success']:
                f.write(f"English PDF: {result['english_pdf']}\n")
                f.write(f"Hebrew PDF: {result['hebrew_pdf']}\n")
                f.write(f"English size: {result['file_sizes'].get('english', 0):,} bytes\n")
                f.write(f"Hebrew size: {result['file_sizes'].get('hebrew', 0):,} bytes\n")
                f.write(f"Generation time: {result['generation_time']:.2f}s\n")
            else:
                f.write(f"Error: {result['error']}\n")
                if result.get('traceback'):
                    f.write(f"Traceback:\n{result['traceback']}\n")
            f.write("-" * 40 + "\n")
    
    print(f"\nDetailed report saved to: {report_file}")
    
    # Final assessment
    if successful_tests == len(test_cases):
        print("\n🎉 ALL TESTS PASSED! PDF generation is robust and working correctly.")
    elif successful_tests >= len(test_cases) * 0.8:
        print("\n⚠️  MOSTLY WORKING: Some edge cases may need attention.")
    else:
        print("\n❌ SIGNIFICANT ISSUES: PDF generation needs fixes.")
    
    return results

if __name__ == "__main__":
    results = main()