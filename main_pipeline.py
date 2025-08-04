"""
Main ML Pipeline for Fraud Detection
Orchestrates the complete ML pipeline from data loading to model serving.
This is the main entry point for training and evaluating all models.
"""

import pandas as pd
import numpy as np
import logging
import os
import time
from datetime import datetime
from typing import Dict, List, Optional

# Import our ML components
from data_pipeline import DataPipeline
from feature_engineering import FeatureEngineer
from quality_scoring import QualityScorer
from traffic_similarity import TrafficSimilarityModel
from anomaly_detection import AnomalyDetector
from model_evaluation import ModelEvaluator
from pdf_report_generator_multilingual import MultilingualPDFReportGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fraud_detection_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class FraudDetectionPipeline:
    """
    Complete ML pipeline for fraud detection and traffic quality scoring.
    """
    
    def __init__(self, data_path: str, output_dir: str = "/home/fiod/shimshi/"):
        self.data_path = data_path
        self.output_dir = output_dir
        self.pipeline_results = {}
        
        # Initialize components
        self.data_pipeline = DataPipeline(data_path)
        self.feature_engineer = FeatureEngineer()
        self.quality_scorer = QualityScorer()
        self.similarity_model = TrafficSimilarityModel()
        self.anomaly_detector = AnomalyDetector()
        self.evaluator = ModelEvaluator()
        self.pdf_generator = PDFReportGenerator(output_dir)
        
    def run_complete_pipeline(self, sample_fraction: float = 0.1) -> Dict:
        """
        Run the complete ML pipeline.
        
        Args:
            sample_fraction: Fraction of data to use (for development/testing)
            
        Returns:
            Dictionary with pipeline results
        """
        logger.info("=" * 60)
        logger.info("STARTING FRAUD DETECTION ML PIPELINE")
        logger.info("=" * 60)
        
        pipeline_start_time = time.time()
        
        try:
            # Step 1: Data Loading and Preprocessing
            logger.info("Step 1: Loading and preprocessing data...")
            step_start = time.time()
            
            df = self.data_pipeline.load_data_chunked(sample_fraction=sample_fraction)
            data_summary = self.data_pipeline.get_data_summary(df)
            quality_report = self.data_pipeline.validate_data_quality(df)
            
            self.pipeline_results['data_loading'] = {
                'records_loaded': len(df),
                'processing_time_seconds': time.time() - step_start,
                'data_summary': data_summary,
                'quality_report': quality_report
            }
            
            logger.info(f"Loaded {len(df):,} records in {time.time() - step_start:.2f} seconds")
            
            # Step 2: Feature Engineering
            logger.info("Step 2: Engineering features...")
            step_start = time.time()
            
            features_df = self.feature_engineer.create_all_features(df)
            channel_features = self.feature_engineer.create_channel_features(features_df)
            
            self.pipeline_results['feature_engineering'] = {
                'original_features': df.shape[1],
                'engineered_features': features_df.shape[1],
                'channel_features': channel_features.shape,
                'processing_time_seconds': time.time() - step_start
            }
            
            logger.info(f"Created {features_df.shape[1] - df.shape[1]} new features in {time.time() - step_start:.2f} seconds")
            
            # Step 3: Quality Scoring Model
            logger.info("Step 3: Training quality scoring model...")
            step_start = time.time()
            
            quality_results_df = self.quality_scorer.score_channels(features_df)
            quality_model_path = os.path.join(self.output_dir, "quality_scoring_model.pkl")
            self.quality_scorer.save_model(quality_model_path)
            
            self.pipeline_results['quality_scoring'] = {
                'channels_scored': len(quality_results_df),
                'score_distribution': quality_results_df['quality_score'].describe().to_dict(),
                'category_distribution': quality_results_df['quality_category'].value_counts().to_dict(),
                'high_risk_channels': quality_results_df['high_risk'].sum(),
                'processing_time_seconds': time.time() - step_start,
                'model_path': quality_model_path
            }
            
            logger.info(f"Quality scoring completed in {time.time() - step_start:.2f} seconds")
            
            # Step 4: Traffic Similarity Model
            logger.info("Step 4: Training traffic similarity model...")
            step_start = time.time()
            
            similarity_results = self.similarity_model.fit(channel_features)
            similarity_model_path = os.path.join(self.output_dir, "traffic_similarity_model.pkl")
            self.similarity_model.save_model(similarity_model_path)
            
            # Get cluster profiles
            cluster_profiles = self.similarity_model.get_cluster_profiles(channel_features)
            outlier_channels = self.similarity_model.detect_outlier_channels(channel_features)
            
            self.pipeline_results['traffic_similarity'] = {
                'clustering_results': similarity_results,
                'cluster_profiles': len(cluster_profiles),
                'outlier_channels': len(outlier_channels),
                'processing_time_seconds': time.time() - step_start,
                'model_path': similarity_model_path
            }
            
            logger.info(f"Similarity modeling completed in {time.time() - step_start:.2f} seconds")
            
            # Step 5: Anomaly Detection
            logger.info("Step 5: Training anomaly detection models...")
            step_start = time.time()
            
            anomaly_results = self.anomaly_detector.run_comprehensive_anomaly_detection(features_df)
            anomaly_model_path = os.path.join(self.output_dir, "anomaly_detection_model.pkl")
            self.anomaly_detector.save_model(anomaly_model_path)
            
            # Count anomalies by type
            anomaly_summary = {}
            if not anomaly_results.empty:
                anomaly_cols = [col for col in anomaly_results.columns if 'anomaly' in col]
                for col in anomaly_cols:
                    if anomaly_results[col].dtype == bool:
                        anomaly_summary[col] = anomaly_results[col].sum()
            
            self.pipeline_results['anomaly_detection'] = {
                'entities_analyzed': len(anomaly_results) if not anomaly_results.empty else 0,
                'anomaly_summary': anomaly_summary,
                'processing_time_seconds': time.time() - step_start,
                'model_path': anomaly_model_path
            }
            
            logger.info(f"Anomaly detection completed in {time.time() - step_start:.2f} seconds")
            
            # Step 6: Model Evaluation
            logger.info("Step 6: Evaluating all models...")
            step_start = time.time()
            
            # Evaluate quality scoring
            quality_metrics = self.evaluator.evaluate_quality_scoring_model(
                self.quality_scorer, features_df
            )
            
            # Evaluate similarity model
            similarity_metrics = self.evaluator.evaluate_similarity_model(
                self.similarity_model, channel_features
            )
            
            # Evaluate anomaly detection
            anomaly_metrics = self.evaluator.evaluate_anomaly_detection_model(
                self.anomaly_detector, features_df
            )
            
            # Cross-validation
            cv_results = self.evaluator.cross_validate_models(
                self.quality_scorer, features_df, cv_folds=3
            )
            
            # Generate evaluation report
            evaluation_report = self.evaluator.generate_evaluation_report()
            
            # Save evaluation results
            evaluation_path = os.path.join(self.output_dir, "model_evaluation_results.pkl")
            self.evaluator.save_evaluation_results(evaluation_path)
            
            self.pipeline_results['model_evaluation'] = {
                'quality_metrics': quality_metrics,
                'similarity_metrics': similarity_metrics,
                'anomaly_metrics': anomaly_metrics,
                'cross_validation': cv_results,
                'evaluation_report': evaluation_report,
                'processing_time_seconds': time.time() - step_start,
                'results_path': evaluation_path
            }
            
            logger.info(f"Model evaluation completed in {time.time() - step_start:.2f} seconds")
            
            # Step 7: Generate Final Results
            logger.info("Step 7: Generating final results...")
            self._generate_final_results(quality_results_df, cluster_profiles, anomaly_results)
            
            # Pipeline summary
            total_time = time.time() - pipeline_start_time
            self.pipeline_results['pipeline_summary'] = {
                'total_processing_time_seconds': total_time,
                'total_processing_time_minutes': total_time / 60,
                'records_processed': len(df),
                'channels_analyzed': len(channel_features),
                'models_trained': 3,
                'completion_status': 'SUCCESS'
            }
            
            # Step 8: Generate Comprehensive RESULTS.md
            logger.info("Step 8: Generating comprehensive RESULTS.md report...")
            self._generate_results_markdown(quality_results_df, cluster_profiles, anomaly_results)
            
            # Step 9: Generate Professional PDF Report
            logger.info("Step 9: Generating comprehensive PDF report...")
            pdf_report_path = self._generate_pdf_report(quality_results_df, cluster_profiles, anomaly_results)
            
            logger.info("=" * 60)
            logger.info("FRAUD DETECTION PIPELINE COMPLETED SUCCESSFULLY")
            logger.info(f"Total time: {total_time/60:.2f} minutes")
            logger.info(f"PDF Report: {pdf_report_path}")
            logger.info("=" * 60)
            
            return self.pipeline_results
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            self.pipeline_results['pipeline_summary'] = {
                'completion_status': 'FAILED',
                'error': str(e),
                'total_processing_time_seconds': time.time() - pipeline_start_time
            }
            raise
    
    def _generate_final_results(self, 
                              quality_results: pd.DataFrame,
                              cluster_profiles: Dict,
                              anomaly_results: pd.DataFrame):
        """Generate final consolidated results."""
        
        # Top quality channels
        top_quality_channels = quality_results.nlargest(20, 'quality_score')[
            ['quality_score', 'quality_category', 'volume', 'bot_rate', 'high_risk']
        ]
        
        # Bottom quality channels  
        bottom_quality_channels = quality_results.nsmallest(20, 'quality_score')[
            ['quality_score', 'quality_category', 'volume', 'bot_rate', 'high_risk']
        ]
        
        # High-risk channels
        high_risk_channels = quality_results[quality_results['high_risk'] == True][
            ['quality_score', 'quality_category', 'volume', 'bot_rate']
        ].head(50)
        
        # Most anomalous channels
        most_anomalous = pd.DataFrame()
        if not anomaly_results.empty and 'overall_anomaly_count' in anomaly_results.columns:
            most_anomalous = anomaly_results.nlargest(20, 'overall_anomaly_count')
        
        # Save consolidated results
        results = {
            'top_quality_channels': top_quality_channels.to_dict('records'),
            'bottom_quality_channels': bottom_quality_channels.to_dict('records'),
            'high_risk_channels': high_risk_channels.to_dict('records'),
            'most_anomalous_channels': most_anomalous.to_dict('records') if not most_anomalous.empty else [],
            'cluster_summary': {
                'total_clusters': len(cluster_profiles),
                'cluster_names': list(cluster_profiles.keys())
            }
        }
        
        # Save to JSON
        import json
        results_path = os.path.join(self.output_dir, "final_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save detailed DataFrames
        quality_results.to_csv(os.path.join(self.output_dir, "channel_quality_scores.csv"))
        if not anomaly_results.empty:
            anomaly_results.to_csv(os.path.join(self.output_dir, "channel_anomaly_scores.csv"))
        
        logger.info(f"Final results saved to {results_path}")
        
        # Log key insights
        logger.info("KEY INSIGHTS:")
        logger.info(f"- Analyzed {len(quality_results):,} channels")
        logger.info(f"- High-risk channels: {len(high_risk_channels):,}")
        logger.info(f"- Average quality score: {quality_results['quality_score'].mean():.2f}")
        logger.info(f"- Quality categories: {dict(quality_results['quality_category'].value_counts())}")
        
        if not anomaly_results.empty and 'overall_anomaly_count' in anomaly_results.columns:
            anomalous_count = (anomaly_results['overall_anomaly_count'] > 0).sum()
            logger.info(f"- Channels with anomalies: {anomalous_count:,}")
    
    def _generate_results_markdown(self,
                                 quality_results: pd.DataFrame,
                                 cluster_profiles: Dict,
                                 anomaly_results: pd.DataFrame):
        """Generate comprehensive RESULTS.md file with TL;DR and detailed analysis."""
        
        # Prepare data
        total_channels = len(quality_results)
        high_risk_channels = quality_results[quality_results['high_risk'] == True]
        anomalous_channels = pd.DataFrame()
        if not anomaly_results.empty and 'overall_anomaly_count' in anomaly_results.columns:
            anomalous_channels = anomaly_results[anomaly_results['overall_anomaly_count'] > 0]
        
        # Quality distribution
        quality_dist = quality_results['quality_category'].value_counts().to_dict()
        
        # Calculate key metrics
        avg_quality_score = quality_results['quality_score'].mean()
        avg_bot_rate = quality_results['bot_rate'].mean()
        total_volume = quality_results['volume'].sum()
        
        # Generate timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Build markdown content
        md_content = f"""# Fraud Detection ML Pipeline Results

Generated: {timestamp}

## TL;DR (Executive Summary)

### 🎯 Key Findings

- **Total Channels Analyzed**: {total_channels:,}
- **High-Risk Channels Identified**: {len(high_risk_channels):,} ({len(high_risk_channels)/total_channels*100:.1f}%)
- **Channels with Anomalies**: {len(anomalous_channels):,} ({len(anomalous_channels)/total_channels*100:.1f}%)
- **Average Quality Score**: {avg_quality_score:.2f}/10
- **Average Bot Rate**: {avg_bot_rate*100:.1f}%
- **Total Traffic Volume**: {total_volume:,} requests

### 🚨 Critical Actions Required

1. **Immediate Investigation**: {len(high_risk_channels)} channels flagged as high-risk require immediate review
2. **Anomaly Review**: {len(anomalous_channels)} channels show suspicious patterns that need verification
3. **Quality Distribution**: {quality_dist.get('Low', 0)} low-quality channels should be considered for removal

### 📊 Quality Distribution

```
High Quality:       {quality_dist.get('High', 0):>4} channels ({quality_dist.get('High', 0)/total_channels*100:>5.1f}%)
Medium-High:        {quality_dist.get('Medium-High', 0):>4} channels ({quality_dist.get('Medium-High', 0)/total_channels*100:>5.1f}%)
Medium-Low:         {quality_dist.get('Medium-Low', 0):>4} channels ({quality_dist.get('Medium-Low', 0)/total_channels*100:>5.1f}%)
Low Quality:        {quality_dist.get('Low', 0):>4} channels ({quality_dist.get('Low', 0)/total_channels*100:>5.1f}%)
```

---

## Detailed Analysis

### 1. Quality Scoring Analysis

The quality scoring model evaluated each channel based on multiple fraud indicators:

#### Top 5 High-Risk Channels
"""
        
        # Add high-risk channels table
        if len(high_risk_channels) > 0:
            top_risk = high_risk_channels.nsmallest(5, 'quality_score')
            md_content += "\n| Channel ID | Quality Score | Bot Rate | Volume | Risk Factors |\n"
            md_content += "|------------|---------------|----------|--------|-------------|\n"
            for _, channel in top_risk.iterrows():
                risk_factors = []
                if channel['bot_rate'] > 0.5:
                    risk_factors.append("High bot rate")
                if channel['volume'] < 10:
                    risk_factors.append("Low volume")
                if channel.get('fraud_score_avg', 0) > 0.5:
                    risk_factors.append("High fraud score")
                
                md_content += f"| {channel['channelId'][:8]}... | {channel['quality_score']:.2f} | {channel['bot_rate']*100:.1f}% | {channel['volume']} | {', '.join(risk_factors)} |\n"
        
        md_content += f"""

#### Top 5 High-Quality Channels
"""
        
        # Add high-quality channels table
        top_quality = quality_results.nlargest(5, 'quality_score')
        md_content += "\n| Channel ID | Quality Score | Bot Rate | Volume | Strengths |\n"
        md_content += "|------------|---------------|----------|--------|----------|\n"
        for _, channel in top_quality.iterrows():
            strengths = []
            if channel['bot_rate'] < 0.05:
                strengths.append("Low bot rate")
            if channel['volume'] > 100:
                strengths.append("High volume")
            if channel.get('ip_diversity', 0) > 0.5:
                strengths.append("Diverse IPs")
            
            md_content += f"| {channel['channelId'][:8]}... | {channel['quality_score']:.2f} | {channel['bot_rate']*100:.1f}% | {channel['volume']} | {', '.join(strengths)} |\n"
        
        # Anomaly Detection Results
        md_content += f"""

### 2. Anomaly Detection Results

The anomaly detection system identified unusual patterns across multiple dimensions:

"""
        
        if not anomaly_results.empty:
            # Count anomalies by type
            anomaly_types = {}
            for col in anomaly_results.columns:
                if 'anomaly' in col and col != 'overall_anomaly_count' and col != 'overall_anomaly_flag':
                    if anomaly_results[col].dtype == bool:
                        anomaly_types[col] = anomaly_results[col].sum()
            
            md_content += "#### Anomaly Type Distribution\n\n"
            for anomaly_type, count in sorted(anomaly_types.items(), key=lambda x: x[1], reverse=True):
                clean_name = anomaly_type.replace('_', ' ').title()
                md_content += f"- **{clean_name}**: {count} channels\n"
            
            # Most anomalous channels
            if 'overall_anomaly_count' in anomaly_results.columns:
                most_anomalous = anomaly_results.nlargest(5, 'overall_anomaly_count')
                if len(most_anomalous) > 0:
                    md_content += "\n#### Most Anomalous Channels\n\n"
                    md_content += "| Channel ID | Anomaly Count | Anomaly Types |\n"
                    md_content += "|------------|---------------|---------------|\n"
                    
                    for _, channel in most_anomalous.iterrows():
                        anomaly_list = []
                        for col in anomaly_results.columns:
                            if 'anomaly' in col and col not in ['overall_anomaly_count', 'overall_anomaly_flag']:
                                if channel.get(col, False):
                                    anomaly_list.append(col.replace('_anomaly', '').replace('_', ' '))
                        
                        md_content += f"| {channel['channelId'][:8]}... | {int(channel['overall_anomaly_count'])} | {', '.join(anomaly_list[:3])}{'...' if len(anomaly_list) > 3 else ''} |\n"
        
        # Traffic Similarity Clusters
        md_content += f"""

### 3. Traffic Similarity Analysis

Channels were grouped into {len(cluster_profiles)} distinct traffic patterns:

"""
        
        for cluster_name, profile in cluster_profiles.items():
            md_content += f"\n#### {cluster_name}\n"
            md_content += f"- **Size**: {profile['size']} channels\n"
            md_content += f"- **Average Quality**: {profile['avg_quality']:.2f}\n"
            
            if 'characteristics' in profile:
                md_content += "- **Key Characteristics**:\n"
                for char, value in profile['characteristics'].items():
                    if isinstance(value, (int, float)):
                        md_content += f"  - {char}: {value:.2f}\n"
                    else:
                        md_content += f"  - {char}: {value}\n"
        
        # Model Performance
        if 'model_evaluation' in self.pipeline_results:
            eval_results = self.pipeline_results['model_evaluation']
            md_content += f"""

### 4. Model Performance Metrics

#### Quality Scoring Model
- **R² Score**: {eval_results.get('quality_metrics', {}).get('r2_score', 'N/A')}
- **Cross-Validation Score**: {eval_results.get('cross_validation', {}).get('quality_cv_score', 'N/A')}

#### Anomaly Detection Performance
- **Total Anomalies Detected**: {len(anomalous_channels)}
- **Detection Coverage**: {len(anomalous_channels)/total_channels*100:.1f}% of channels

#### Traffic Similarity Model
- **Silhouette Score**: {eval_results.get('similarity_metrics', {}).get('silhouette_score', 'N/A')}
- **Number of Clusters**: {len(cluster_profiles)}
"""
        
        # Recommendations
        md_content += f"""

### 5. Recommendations

Based on the analysis, we recommend the following actions:

#### 🔴 Immediate Actions (High Priority)

1. **Block/Investigate High-Risk Channels**
   - {len(high_risk_channels)} channels identified as high-risk
   - Average bot rate in this group: {high_risk_channels['bot_rate'].mean()*100:.1f}%
   - Potential revenue at risk: ${high_risk_channels['volume'].sum() * 0.1:.2f} (estimated)

2. **Review Anomalous Patterns**
   - {len(anomalous_channels)} channels show unusual behavior patterns
   - Focus on channels with multiple anomaly types
   - Verify legitimacy through manual review

#### 🟡 Short-term Actions (Medium Priority)

1. **Quality Improvement**
   - Work with {quality_dist.get('Medium-Low', 0)} medium-low quality channels
   - Implement stricter traffic filtering
   - Monitor improvement over 30 days

2. **Pattern Monitoring**
   - Set up alerts for channels matching high-risk patterns
   - Track quality score changes weekly
   - Monitor for new anomaly patterns

#### 🟢 Long-term Actions (Low Priority)

1. **Model Enhancement**
   - Retrain models monthly with new data
   - Add new fraud indicators as discovered
   - Improve anomaly detection sensitivity

2. **Process Optimization**
   - Automate channel blocking for scores < 2.0
   - Implement real-time scoring for new channels
   - Create dashboard for ongoing monitoring

---

## Technical Details

### Pipeline Configuration
- **Data Source**: {self.data_path}
- **Processing Time**: {self.pipeline_results['pipeline_summary']['total_processing_time_minutes']:.1f} minutes
- **Records Processed**: {self.pipeline_results['pipeline_summary']['records_processed']:,}
- **Models Trained**: {self.pipeline_results['pipeline_summary']['models_trained']}

### Feature Engineering
- **Total Features Created**: {len(self.pipeline_results.get('feature_engineering', {}).get('feature_names', []))}
- **Feature Categories**: Temporal, Geographic, Behavioral, Device, Volume

### Output Files Generated
- `channel_quality_scores.csv` - Detailed quality scores for all channels
- `channel_anomaly_scores.csv` - Anomaly detection results
- `final_results.json` - Machine-readable results summary
- `RESULTS.md` - This comprehensive report

---

*Report generated by Fraud Detection ML Pipeline v1.0*
"""
        
        # Save the markdown file
        results_path = os.path.join(self.output_dir, "RESULTS.md")
        with open(results_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
        
        logger.info(f"Comprehensive results report saved to {results_path}")
    
    def _generate_pdf_report(self,
                           quality_results: pd.DataFrame,
                           cluster_profiles: Dict,
                           anomaly_results: pd.DataFrame) -> str:
        """Generate comprehensive PDF report with all visualizations and insights."""
        
        try:
            # Load final results for comprehensive data
            final_results = {}
            final_results_path = os.path.join(self.output_dir, "final_results.json")
            if os.path.exists(final_results_path):
                import json
                with open(final_results_path, 'r') as f:
                    final_results = json.load(f)
            
            # Generate the PDF report
            pdf_path = self.pdf_generator.generate_comprehensive_report(
                quality_results, 
                anomaly_results, 
                final_results, 
                self.pipeline_results
            )
            
            logger.info(f"PDF report generated successfully: {pdf_path}")
            return pdf_path
            
        except Exception as e:
            logger.error(f"Failed to generate PDF report: {e}")
            # Don't fail the entire pipeline if PDF generation fails
            return f"PDF generation failed: {str(e)}"

def main():
    """Main execution function."""
    # Configuration
    DATA_PATH = "/home/fiod/shimshi/bq-results-20250804-141411-1754316868932.csv"
    OUTPUT_DIR = "/home/fiod/shimshi/"
    SAMPLE_FRACTION = 0.2  # Use 1% of data for demo (adjust as needed)
    
    # Initialize and run pipeline
    pipeline = FraudDetectionPipeline(DATA_PATH, OUTPUT_DIR)
    
    try:
        results = pipeline.run_complete_pipeline(sample_fraction=SAMPLE_FRACTION)
        
        # Print summary
        summary = results.get('pipeline_summary', {})
        print("\nPIPELINE SUMMARY:")
        print(f"Status: {summary.get('completion_status', 'Unknown')}")
        print(f"Processing time: {summary.get('total_processing_time_minutes', 0):.1f} minutes")
        print(f"Records processed: {summary.get('records_processed', 0):,}")
        print(f"Channels analyzed: {summary.get('channels_analyzed', 0):,}")
        
        return results
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        raise

if __name__ == "__main__":
    results = main()
