"""
Test script for multilingual PDF generation
"""

import pandas as pd
import numpy as np
from pdf_report_generator_multilingual import MultilingualPDFReportGenerator
import logging

logging.basicConfig(level=logging.INFO)

# Create sample data
np.random.seed(42)
n_channels = 1000

quality_df = pd.DataFrame({
    'channelId': [f'CH{i:04d}' for i in range(n_channels)],
    'quality_score': np.random.uniform(0, 10, n_channels),
    'bot_rate': np.random.beta(2, 5, n_channels),
    'volume': np.random.lognormal(5, 2, n_channels),
    'quality_category': pd.cut(np.random.uniform(0, 10, n_channels), 
                              bins=[0, 2.5, 5, 7.5, 10], 
                              labels=['Low', 'Medium-Low', 'Medium-High', 'High']),
    'high_risk': np.random.choice([True, False], n_channels, p=[0.2, 0.8])
})

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
