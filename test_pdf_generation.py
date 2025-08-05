"""
Test script for multilingual PDF generation
"""

import pandas as pd
import numpy as np
from pdf_report_generator_multilingual import MultilingualPDFReportGenerator
import logging

logging.basicConfig(level=logging.INFO)

# Create sample data - test with small dataset that would cause original errors
np.random.seed(42)

# Test case: Small dataset with limited unique values (edge case that caused original errors)
n_channels = 5  # Very small dataset

quality_df = pd.DataFrame({
    'channelId': [f'CH{i:04d}' for i in range(n_channels)],
    'quality_score': [3.0, 3.0, 7.0, 7.0, 7.0],  # Only 2 unique values
    'bot_rate': [0.1, 0.1, 0.8, 0.8, 0.8],       # Only 2 unique values  
    'volume': [50, 50, 200, 200, 200],           # Only 2 unique values
    'quality_category': ['Low', 'Low', 'High', 'High', 'High'],
    'high_risk': [False, False, True, True, True]
})

print("TESTING WITH EDGE CASE DATA:")
print(f"Dataset size: {len(quality_df)} rows")
print(f"Unique quality scores: {quality_df['quality_score'].nunique()}")
print(f"Unique bot rates: {quality_df['bot_rate'].nunique()}")
print(f"Unique volumes: {quality_df['volume'].nunique()}")
print("This data would have caused 'bins must increase monotonically' errors in the original code.")

# Create sample anomaly data
anomaly_df = pd.DataFrame({
    'channelId': quality_df['channelId'].sample(500),
    'temporal_anomaly': np.random.choice([True, False], 500, p=[0.3, 0.7]),
    'geographic_anomaly': np.random.choice([True, False], 500, p=[0.2, 0.8]),
    'volume_anomaly': np.random.choice([True, False], 500, p=[0.15, 0.85]),
    'device_anomaly': np.random.choice([True, False], 500, p=[0.25, 0.75]),
    'behavioral_anomaly': np.random.choice([True, False], 500, p=[0.2, 0.8])
})

# Calculate overall anomaly count
anomaly_cols = [col for col in anomaly_df.columns if 'anomaly' in col]
anomaly_df['overall_anomaly_count'] = anomaly_df[anomaly_cols].sum(axis=1)

# Create sample results
final_results = {
    'top_quality_channels': quality_df.nlargest(10, 'quality_score').to_dict('records'),
    'high_risk_channels': quality_df[quality_df['high_risk'] == True].head(10).to_dict('records'),
    'cluster_summary': {
        'total_clusters': 5,
        'cluster_names': ['Cluster 0', 'Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4']
    }
}

pipeline_results = {
    'pipeline_summary': {
        'total_processing_time_minutes': 5.2,
        'records_processed': len(quality_df),
        'channels_analyzed': len(quality_df),
        'models_trained': 3
    },
    'model_evaluation': {
        'quality_metrics': {'r2_score': 0.85},
        'cross_validation': {'quality_cv_score': 0.82},
        'similarity_metrics': {'silhouette_score': 0.65}
    },
    'feature_engineering': {
        'feature_names': ['feature1', 'feature2', 'feature3']
    }
}

# Generate reports
print("Generating multilingual PDF reports...")
generator = MultilingualPDFReportGenerator()
en_path, he_path = generator.generate_comprehensive_report(
    quality_df, 
    anomaly_df, 
    final_results, 
    pipeline_results
)

print(f"\nEnglish PDF generated: {en_path}")
print(f"Hebrew PDF generated: {he_path}")
print("\nTest completed successfully!")
