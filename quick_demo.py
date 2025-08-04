#!/usr/bin/env python3
"""
Quick Demo of Traffic Analysis Pipeline
Runs the core analysis pipeline on a small sample for demonstration.
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime

# Import our ML components
from data_pipeline import DataPipeline
from feature_engineering import FeatureEngineer
from quality_scoring import QualityScorer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Quick demo of the traffic analysis pipeline."""
    
    print("=" * 60)
    print("TRAFFIC QUALITY ANALYSIS - QUICK DEMO")
    print("=" * 60)
    
    # Configuration
    DATA_PATH = "/home/fiod/shimshi/bq-results-20250804-141411-1754316868932.csv"
    SAMPLE_FRACTION = 0.005  # Use 0.5% of data for quick demo
    
    try:
        # Step 1: Load data
        print(f"\n🔄 Step 1: Loading {SAMPLE_FRACTION*100:.1f}% sample of data...")
        data_pipeline = DataPipeline(DATA_PATH, chunk_size=5000)
        df = data_pipeline.load_data_chunked(sample_fraction=SAMPLE_FRACTION)
        print(f"✅ Loaded {len(df):,} records")
        
        # Step 2: Feature engineering
        print(f"\n🔄 Step 2: Engineering features...")
        feature_engineer = FeatureEngineer()
        features_df = feature_engineer.create_all_features(df)
        print(f"✅ Created {features_df.shape[1] - df.shape[1]} new features")
        
        # Step 3: Quality scoring
        print(f"\n🔄 Step 3: Scoring traffic quality...")
        quality_scorer = QualityScorer()
        quality_results = quality_scorer.score_channels(features_df)
        print(f"✅ Scored {len(quality_results)} channels")
        
        # Results summary
        print(f"\n" + "=" * 60)
        print("📊 ANALYSIS RESULTS")
        print("=" * 60)
        
        # Quality distribution
        quality_dist = quality_results['quality_category'].value_counts()
        print(f"Quality Distribution:")
        for category, count in quality_dist.items():
            print(f"  {category}: {count} channels")
        
        # Show available columns for debugging
        print(f"\nAvailable columns: {list(quality_results.columns)}")
        
        # Use the correct column name for scoring
        score_col = 'quality_score' if 'quality_score' in quality_results.columns else quality_results.columns[1]
        
        # Top and bottom channels (channelId is likely the index)
        top_channels = quality_results.nlargest(5, score_col)
        bottom_channels = quality_results.nsmallest(5, score_col)
        
        print(f"\n🏆 Top 5 Quality Channels:")
        for channel_id, row in top_channels.iterrows():
            print(f"  {channel_id[:8]}... - Score: {row[score_col]:.2f} ({row['quality_category']})")
        
        print(f"\n⚠️  Bottom 5 Quality Channels:")
        for channel_id, row in bottom_channels.iterrows():
            print(f"  {channel_id[:8]}... - Score: {row[score_col]:.2f} ({row['quality_category']})")
        
        # Basic traffic stats
        print(f"\n📈 Traffic Statistics:")
        print(f"  Total Records: {len(df):,}")
        print(f"  Unique Channels: {df['channelId'].nunique()}")
        print(f"  Unique Publishers: {df['publisherId'].nunique()}")
        print(f"  Countries: {df['country'].nunique()}")
        print(f"  Bot Traffic: {df['isLikelyBot'].sum():,} ({df['isLikelyBot'].mean()*100:.1f}%)")
        print(f"  Mobile Traffic: {(df['device'] == 'mobile').sum():,} ({(df['device'] == 'mobile').mean()*100:.1f}%)")
        
        # IP Risk analysis
        print(f"\n🔍 IP Risk Analysis:")
        high_risk_ips = (df['isIpDatacenter'] | df['isIpVPN'] | df['isIpTOR'] | df['isIpPublicProxy']).sum()
        print(f"  High-risk IPs: {high_risk_ips:,} ({high_risk_ips/len(df)*100:.1f}%)")
        
        print(f"\n✅ Demo completed successfully!")
        print(f"💾 Results saved to quality scoring files in current directory")
        
        return quality_results
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise

if __name__ == "__main__":
    results = main()