#!/usr/bin/env python3
"""
Test script to verify that all original columns are preserved in the fraud classification output
"""

import pandas as pd
import numpy as np
import sys
import os
sys.path.append('/home/fiod/shimshi')

def create_test_csv():
    """Create a small test CSV with all 30 original columns"""
    np.random.seed(42)
    n_rows = 100
    
    # Match the exact structure of the original CSV
    data = {
        'date': pd.date_range('2025-08-01', periods=n_rows, freq='h'),
        'keyword': np.random.choice(['keyword1', 'keyword2', 'keyword3'], n_rows),
        'country': np.random.choice(['US', 'CA', 'UK'], n_rows),
        'browser': np.random.choice(['chrome', 'firefox', 'safari'], n_rows),
        'device': np.random.choice(['mobile', 'desktop', 'tablet'], n_rows),
        'referrer': np.random.choice(['google.com', 'facebook.com', None], n_rows),  # THIS WAS MISSING
        'ip': [f'192.168.{np.random.randint(1,255)}.{np.random.randint(1,255)}' for _ in range(n_rows)],
        'publisherId': [f'pub_{i%5}' for i in range(n_rows)],
        'channelId': [f'ch_{i%10}' for i in range(n_rows)],
        'advertiserId': [f'adv_{i%3}' for i in range(n_rows)],
        'feedId': [f'feed_{i%4}' for i in range(n_rows)],
        'browserMajorVersion': np.random.choice([90, 91, 92, 93, 94], n_rows),  # THIS WAS MISSING
        'userId': [f'user_{i%20}' for i in range(n_rows)],
        'isLikelyBot': np.random.choice([True, False], n_rows),
        'ipClassification': np.random.choice(['clean', 'suspicious'], n_rows),
        'isIpDatacenter': np.random.choice([True, False], n_rows),
        'datacenterName': np.random.choice(['AWS', 'Google', None], n_rows),  # THIS WAS MISSING
        'ipHostName': [f'host_{i}.com' for i in range(n_rows)],  # THIS WAS MISSING
        'isIpAnonymous': np.random.choice([True, False], n_rows),
        'isIpCrawler': np.random.choice([True, False], n_rows),  # THIS WAS MISSING
        'isIpPublicProxy': np.random.choice([True, False], n_rows),  # THIS WAS MISSING
        'isIpVPN': np.random.choice([True, False], n_rows),  # THIS WAS MISSING
        'isIpHostingService': np.random.choice([True, False], n_rows),  # THIS WAS MISSING
        'isIpTOR': np.random.choice([True, False], n_rows),  # THIS WAS MISSING
        'isIpResidentialProxy': np.random.choice([True, False], n_rows),  # THIS WAS MISSING
        'performance': np.random.uniform(0.1, 2.0, n_rows),  # THIS WAS MISSING
        'detection': np.random.choice(['detected', 'undetected'], n_rows),  # THIS WAS MISSING
        'platform': np.random.choice(['web', 'mobile', 'api'], n_rows),  # THIS WAS MISSING
        'location': np.random.choice(['US-CA', 'US-NY', 'CA-ON'], n_rows),  # THIS WAS MISSING
        'userAgent': [f'Mozilla/5.0 Agent {i}' for i in range(n_rows)]  # THIS WAS MISSING
    }
    
    df = pd.DataFrame(data)
    csv_path = '/home/fiod/shimshi/test_all_columns.csv'
    df.to_csv(csv_path, index=False)
    
    print(f"Created test CSV with ALL 30 original columns:")
    print(f"  File: {csv_path}")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {len(df.columns)}")
    
    return csv_path, df.columns.tolist()

def test_column_preservation():
    """Test that fraud classification preserves all original columns"""
    print("=== COLUMN PRESERVATION TEST ===")
    
    # Create test data
    csv_path, original_columns = create_test_csv()
    
    try:
        from main_pipeline_optimized import OptimizedFraudDetectionPipeline
        
        print(f"\nOriginal CSV columns ({len(original_columns)}):")
        print(", ".join(original_columns))
        
        # Run the optimized pipeline
        pipeline = OptimizedFraudDetectionPipeline(
            data_path=csv_path,
            output_dir='/home/fiod/shimshi/',
            n_jobs=2,
            approximate=True,
            sample_fraction=1.0
        )
        
        print(f"\nRunning optimized pipeline...")
        results = pipeline.run_complete_pipeline()
        
        # Check if fraud classification file exists
        fraud_csv_path = '/home/fiod/shimshi/fraud_classification_results.csv'
        if os.path.exists(fraud_csv_path):
            # Read the fraud classification results
            fraud_df = pd.read_csv(fraud_csv_path)
            fraud_columns = fraud_df.columns.tolist()
            
            print(f"\nFraud classification CSV columns ({len(fraud_columns)}):")
            print(", ".join(fraud_columns))
            
            # Identify original columns that are preserved
            preserved_original = [col for col in original_columns if col in fraud_columns]
            missing_original = [col for col in original_columns if col not in fraud_columns]
            
            # Identify new classification columns
            new_classification_cols = [col for col in fraud_columns if col not in original_columns]
            
            print(f"\n=== ANALYSIS ===")
            print(f"Original columns: {len(original_columns)}")
            print(f"Preserved original columns: {len(preserved_original)}")
            print(f"Missing original columns: {len(missing_original)}")
            print(f"New classification columns: {len(new_classification_cols)}")
            print(f"Total fraud CSV columns: {len(fraud_columns)}")
            
            if missing_original:
                print(f"\n❌ MISSING ORIGINAL COLUMNS:")
                for col in missing_original:
                    print(f"  - {col}")
            else:
                print(f"\n✅ ALL ORIGINAL COLUMNS PRESERVED!")
            
            if new_classification_cols:
                print(f"\n✅ NEW CLASSIFICATION COLUMNS ADDED:")
                for col in new_classification_cols:
                    print(f"  - {col}")
            
            # Check file sizes
            original_size = os.path.getsize(csv_path)
            fraud_size = os.path.getsize(fraud_csv_path)
            
            print(f"\n=== FILE SIZE COMPARISON ===")
            print(f"Original CSV: {original_size:,} bytes")
            print(f"Fraud CSV: {fraud_size:,} bytes")
            
            if fraud_size > original_size:
                print(f"✅ SUCCESS: Fraud CSV is larger ({fraud_size - original_size:,} bytes more)")
            else:
                print(f"❌ ISSUE: Fraud CSV is smaller ({original_size - fraud_size:,} bytes less)")
            
            # Success criteria
            success = (
                len(missing_original) == 0 and  # All original columns preserved
                len(new_classification_cols) > 0 and  # New columns added
                fraud_size > original_size  # File is larger
            )
            
            return success
        else:
            print(f"❌ Fraud classification file not found: {fraud_csv_path}")
            return False
            
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Clean up test file
        if os.path.exists(csv_path):
            os.remove(csv_path)
            print(f"\nCleaned up test file: {csv_path}")

def main():
    success = test_column_preservation()
    
    print(f"\n{'='*50}")
    if success:
        print("🎉 SUCCESS: All original columns are preserved in fraud classification!")
        print("The fraud classification CSV should now be larger than the original.")
    else:
        print("⚠️  ISSUE: Some original columns are still missing or other problems exist.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)