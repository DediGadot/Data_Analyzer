"""Test RESULTS.md generation with mock data"""

import pandas as pd
import numpy as np
from datetime import datetime
import os

# Create mock quality results
quality_results = pd.DataFrame({
    'channelId': [f'ch_{i:04d}' for i in range(100)],
    'quality_score': np.random.uniform(1, 10, 100),
    'quality_category': np.random.choice(['High', 'Medium-High', 'Medium-Low', 'Low'], 100, p=[0.3, 0.5, 0.15, 0.05]),
    'volume': np.random.randint(1, 1000, 100),
    'bot_rate': np.random.uniform(0, 1, 100),
    'high_risk': np.random.choice([True, False], 100, p=[0.1, 0.9]),
    'fraud_score_avg': np.random.uniform(0, 1, 100),
    'ip_diversity': np.random.uniform(0, 1, 100)
})

# Create mock anomaly results
anomaly_results = pd.DataFrame({
    'channelId': quality_results['channelId'],
    'temporal_anomaly': np.random.choice([True, False], 100, p=[0.2, 0.8]),
    'geographic_anomaly': np.random.choice([True, False], 100, p=[0.15, 0.85]),
    'device_anomaly': np.random.choice([True, False], 100, p=[0.1, 0.9]),
    'behavioral_anomaly': np.random.choice([True, False], 100, p=[0.25, 0.75]),
    'volume_anomaly': np.random.choice([True, False], 100, p=[0.1, 0.9])
})

# Calculate overall anomaly count
anomaly_cols = [col for col in anomaly_results.columns if 'anomaly' in col]
anomaly_results['overall_anomaly_count'] = anomaly_results[anomaly_cols].sum(axis=1)
anomaly_results['overall_anomaly_flag'] = anomaly_results['overall_anomaly_count'] >= 2

# Create mock cluster profiles
cluster_profiles = {
    'High-Volume Quality Traffic': {
        'size': 30,
        'avg_quality': 8.5,
        'characteristics': {
            'avg_volume': 500,
            'avg_bot_rate': 0.02,
            'ip_diversity': 0.85
        }
    },
    'Medium-Volume Mixed Traffic': {
        'size': 50,
        'avg_quality': 6.2,
        'characteristics': {
            'avg_volume': 150,
            'avg_bot_rate': 0.15,
            'ip_diversity': 0.6
        }
    },
    'Low-Volume Suspicious Traffic': {
        'size': 20,
        'avg_quality': 3.1,
        'characteristics': {
            'avg_volume': 25,
            'avg_bot_rate': 0.75,
            'ip_diversity': 0.2
        }
    }
}

# Create mock pipeline results
pipeline_results = {
    'pipeline_summary': {
        'total_processing_time_minutes': 2.5,
        'records_processed': 10000,
        'models_trained': 3
    },
    'feature_engineering': {
        'feature_names': ['feature_' + str(i) for i in range(67)]
    },
    'model_evaluation': {
        'quality_metrics': {'r2_score': 0.85},
        'cross_validation': {'quality_cv_score': 0.82},
        'similarity_metrics': {'silhouette_score': 0.65}
    }
}

# Now generate the RESULTS.md using the method from main_pipeline
print("Generating RESULTS.md with mock data...")

# Import and instantiate pipeline just for the markdown generation
from main_pipeline import FraudDetectionPipeline

pipeline = FraudDetectionPipeline("/dummy/path", ".")
pipeline.pipeline_results = pipeline_results
pipeline._generate_results_markdown(quality_results, cluster_profiles, anomaly_results)

# Check if file was created
if os.path.exists("RESULTS.md"):
    print("✅ RESULTS.md successfully generated!")
    
    # Show first 100 lines
    with open("RESULTS.md", "r") as f:
        lines = f.readlines()
        print(f"\nShowing first 100 lines of {len(lines)} total lines:\n")
        print("".join(lines[:100]))
else:
    print("❌ RESULTS.md was not generated")