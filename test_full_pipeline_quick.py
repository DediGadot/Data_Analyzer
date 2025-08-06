#!/usr/bin/env python3
"""
Quick test of the full optimized pipeline with synthetic data
"""

import pandas as pd
import numpy as np
import sys
import os
sys.path.append('/home/fiod/shimshi')

def create_synthetic_csv():
    """Create a small synthetic CSV file that matches the real structure"""
    np.random.seed(42)
    
    # Create 1000 rows of synthetic data
    n_rows = 1000
    data = {
        'date': pd.date_range('2025-08-01', periods=n_rows, freq='H'),
        'keyword': np.random.choice(['keyword1', 'keyword2', 'keyword3', 'keyword4'], n_rows),
        'country': np.random.choice(['US', 'CA', 'UK', 'DE', 'FR'], n_rows),
        'browser': np.random.choice(['chrome', 'firefox', 'safari', 'edge'], n_rows),
        'device': np.random.choice(['mobile', 'desktop', 'tablet'], n_rows),
        'referrer': np.random.choice(['google.com', 'facebook.com', 'direct', 'twitter.com'], n_rows),
        'ip': [f'192.168.{np.random.randint(1,255)}.{np.random.randint(1,255)}' for _ in range(n_rows)],
        'publisherId': [f'pub_{i%10}' for i in range(n_rows)],
        'channelId': [f'ch_{i%50}' for i in range(n_rows)],
        'advertiserId': [f'adv_{i%5}' for i in range(n_rows)],
        'feedId': [f'feed_{i%8}' for i in range(n_rows)],
        'browserMajorVersion': np.random.choice([90, 91, 92, 93, 94, 95], n_rows),
        'userId': [f'user_{i%100}' for i in range(n_rows)],  # Correct column name
        'isLikelyBot': np.random.choice([True, False], n_rows, p=[0.2, 0.8]),
        'ipClassification': np.random.choice(['clean', 'suspicious', 'unknown'], n_rows, p=[0.7, 0.2, 0.1]),
        'isIpDatacenter': np.random.choice([True, False], n_rows, p=[0.1, 0.9]),
        'datacenterName': np.random.choice(['AWS', 'Google', 'Azure', None], n_rows, p=[0.05, 0.03, 0.02, 0.9]),
        'ipHostName': [f'host{i}.example.com' if np.random.random() > 0.3 else None for i in range(n_rows)],
        'isIpAnonymous': np.random.choice([True, False], n_rows, p=[0.1, 0.9]),
        'isIpCrawler': np.random.choice([True, False], n_rows, p=[0.05, 0.95]),
        'isIpPublicProxy': np.random.choice([True, False], n_rows, p=[0.08, 0.92]),
        'isIpVPN': np.random.choice([True, False], n_rows, p=[0.12, 0.88]),
        'isIpHostingService': np.random.choice([True, False], n_rows, p=[0.06, 0.94]),
        'isIpTOR': np.random.choice([True, False], n_rows, p=[0.02, 0.98]),
        'isIpResidentialProxy': np.random.choice([True, False], n_rows, p=[0.04, 0.96]),
        'performance': np.random.uniform(0.1, 3.0, n_rows),
        'detection': np.random.choice(['detected', 'undetected', 'flagged'], n_rows, p=[0.6, 0.3, 0.1]),
        'platform': np.random.choice(['web', 'mobile', 'api', 'embedded'], n_rows, p=[0.4, 0.3, 0.2, 0.1]),
        'location': np.random.choice(['US-CA', 'US-NY', 'CA-ON', 'UK-LDN', 'DE-BER'], n_rows),
        'userAgent': [f'Mozilla/5.0 (Platform; Agent {i%20}) AppleWebKit/537.36' for i in range(n_rows)]
    }
    
    df = pd.DataFrame(data)
    
    # Save to CSV
    csv_path = '/home/fiod/shimshi/test_synthetic_data.csv'
    df.to_csv(csv_path, index=False)
    print(f"Created synthetic test data: {csv_path}")
    print(f"Data shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    return csv_path

def main():
    print("=== FULL PIPELINE QUICK TEST ===")
    
    # Create synthetic data
    csv_path = create_synthetic_csv()
    
    # Run the optimized pipeline
    print("\nRunning optimized pipeline...")
    
    try:
        from main_pipeline_optimized import OptimizedFraudDetectionPipeline
        
        pipeline = OptimizedFraudDetectionPipeline(
            data_path=csv_path,
            output_dir='/home/fiod/shimshi/',
            n_jobs=4,
            approximate=True,
            sample_fraction=1.0  # Use all synthetic data
        )
        
        results = pipeline.run_complete_pipeline()
        
        print("✅ PIPELINE COMPLETED SUCCESSFULLY!")
        print("\nResults summary:")
        summary = results.get('pipeline_summary', {})
        print(f"  Status: {summary.get('completion_status', 'Unknown')}")
        print(f"  Processing time: {summary.get('total_processing_time_minutes', 0):.2f} minutes")
        print(f"  Records processed: {summary.get('records_processed', 0):,}")
        
        # Check if classification file was created
        if os.path.exists('/home/fiod/shimshi/fraud_classification_results.csv'):
            print("✅ Classification file created successfully!")
            
            # Quick check of the classification results
            df_results = pd.read_csv('/home/fiod/shimshi/fraud_classification_results.csv')
            print(f"  Classification results: {df_results.shape[0]} rows")
            
            if 'classification' in df_results.columns:
                fraud_count = (df_results['classification'] == 'fraud').sum()
                good_count = (df_results['classification'] == 'good_account').sum()
                print(f"  Fraud accounts: {fraud_count}")
                print(f"  Good accounts: {good_count}")
        
        return True
        
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

if __name__ == "__main__":
    success = main()
    print(f"\n{'SUCCESS' if success else 'FAILURE'}: Pipeline test completed.")
    sys.exit(0 if success else 1)