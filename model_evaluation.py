"""
Model Evaluation and Validation Framework
Comprehensive evaluation system for fraud detection and quality scoring models.
Includes performance metrics, validation strategies, and model monitoring.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score, silhouette_score,
    confusion_matrix, classification_report
)
from sklearn.model_selection import cross_val_score, StratifiedKFold, TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional, Any
import joblib
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class ModelEvaluator:
    """
    Comprehensive model evaluation and validation framework for fraud detection models.
    Provides metrics, visualizations, and monitoring capabilities.
    """
    
    def __init__(self):
        self.evaluation_results = {}
        self.baseline_metrics = {}
        self.model_history = []
        
    def evaluate_quality_scoring_model(self, 
                                     quality_scorer,
                                     features_df: pd.DataFrame,
                                     true_labels: Optional[pd.Series] = None) -> Dict:
        """
        Evaluate quality scoring model performance.
        
        Args:
            quality_scorer: Trained quality scoring model
            features_df: Feature DataFrame
            true_labels: Optional ground truth labels for validation
            
        Returns:
            Dictionary with evaluation metrics
        """
        logger.info("Evaluating quality scoring model")
        
        # Get model predictions
        results_df = quality_scorer.score_channels(features_df)
        predicted_scores = results_df['quality_score']
        
        evaluation_metrics = {
            'model_type': 'quality_scoring',
            'n_channels': len(results_df),
            'score_distribution': {
                'mean': predicted_scores.mean(),
                'std': predicted_scores.std(),
                'min': predicted_scores.min(),
                'max': predicted_scores.max(),
                'quartiles': predicted_scores.quantile([0.25, 0.5, 0.75]).to_dict()
            }
        }
        
        # Category distribution
        if 'quality_category' in results_df.columns:
            category_dist = results_df['quality_category'].value_counts().to_dict()
            evaluation_metrics['category_distribution'] = category_dist
        
        # High-risk channel analysis
        if 'high_risk' in results_df.columns:
            high_risk_count = results_df['high_risk'].sum()
            evaluation_metrics['high_risk_channels'] = {
                'count': high_risk_count,
                'percentage': high_risk_count / len(results_df) * 100
            }
        
        # If ground truth labels are available
        if true_labels is not None:
            # Align predictions with true labels
            common_channels = predicted_scores.index.intersection(true_labels.index)
            if len(common_channels) > 0:
                aligned_pred = predicted_scores[common_channels]
                aligned_true = true_labels[common_channels]
                
                # Regression metrics
                evaluation_metrics['regression_metrics'] = {
                    'mse': mean_squared_error(aligned_true, aligned_pred),
                    'mae': mean_absolute_error(aligned_true, aligned_pred),
                    'r2': r2_score(aligned_true, aligned_pred),
                    'correlation': aligned_true.corr(aligned_pred)
                }
                
                # Classification metrics (convert to categories)
                true_categories = pd.cut(aligned_true, bins=[0, 3, 5, 7, 10], 
                                       labels=['Low', 'Medium-Low', 'Medium-High', 'High'])
                pred_categories = pd.cut(aligned_pred, bins=[0, 3, 5, 7, 10], 
                                       labels=['Low', 'Medium-Low', 'Medium-High', 'High'])
                
                evaluation_metrics['classification_metrics'] = {
                    'accuracy': accuracy_score(true_categories, pred_categories),
                    'precision_macro': precision_score(true_categories, pred_categories, average='macro'),
                    'recall_macro': recall_score(true_categories, pred_categories, average='macro'),
                    'f1_macro': f1_score(true_categories, pred_categories, average='macro')
                }
        
        # Feature importance analysis
        if hasattr(quality_scorer, 'get_feature_importance'):
            feature_importance = quality_scorer.get_feature_importance()
            evaluation_metrics['feature_importance'] = feature_importance
        
        self.evaluation_results['quality_scoring'] = evaluation_metrics
        return evaluation_metrics
    
    def evaluate_similarity_model(self, 
                                similarity_model,
                                channel_features: pd.DataFrame) -> Dict:
        """
        Evaluate traffic similarity model performance.
        
        Args:
            similarity_model: Trained similarity model
            channel_features: Channel features DataFrame
            
        Returns:
            Dictionary with evaluation metrics
        """
        logger.info("Evaluating traffic similarity model")
        
        evaluation_metrics = {
            'model_type': 'traffic_similarity',
            'n_channels': len(channel_features)
        }
        
        # Clustering evaluation metrics
        if hasattr(similarity_model, 'models') and 'kmeans' in similarity_model.models:
            X = similarity_model.prepare_features(channel_features)
            X_scaled = similarity_model.scaler.transform(X)
            
            # Silhouette score
            kmeans_labels = similarity_model.models['kmeans'].labels_
            silhouette_avg = silhouette_score(X_scaled, kmeans_labels)
            
            evaluation_metrics['clustering_metrics'] = {
                'silhouette_score': silhouette_avg,
                'n_clusters': len(np.unique(kmeans_labels)),
                'inertia': similarity_model.models['kmeans'].inertia_
            }
            
            # Cluster size distribution
            cluster_sizes = pd.Series(kmeans_labels).value_counts().sort_index()
            evaluation_metrics['cluster_distribution'] = {
                'sizes': cluster_sizes.to_dict(),
                'size_stats': {
                    'mean': cluster_sizes.mean(),
                    'std': cluster_sizes.std(),
                    'min': cluster_sizes.min(),
                    'max': cluster_sizes.max()
                }
            }
        
        # Outlier detection evaluation
        if hasattr(similarity_model, 'detect_outlier_channels'):
            outliers = similarity_model.detect_outlier_channels(channel_features)
            evaluation_metrics['outlier_detection'] = {
                'n_outliers': len(outliers),
                'outlier_percentage': len(outliers) / len(channel_features) * 100,
                'outlier_channels': outliers[:10]  # Top 10 for inspection
            }
        
        # Embedding quality (if available)
        if hasattr(similarity_model, 'embeddings'):
            embeddings_info = {}
            for embed_type, embedding in similarity_model.embeddings.items():
                if embedding is not None:
                    embeddings_info[embed_type] = {
                        'shape': embedding.shape,
                        'variance_explained': np.var(embedding, axis=0).sum() if embed_type == 'pca' else None
                    }
            evaluation_metrics['embeddings'] = embeddings_info
        
        self.evaluation_results['traffic_similarity'] = evaluation_metrics
        return evaluation_metrics
    
    def evaluate_anomaly_detection_model(self, 
                                       anomaly_detector,
                                       features_df: pd.DataFrame,
                                       known_anomalies: Optional[List[str]] = None) -> Dict:
        """
        Evaluate anomaly detection model performance.
        
        Args:
            anomaly_detector: Trained anomaly detection model
            features_df: Feature DataFrame
            known_anomalies: Optional list of known anomalous channel IDs
            
        Returns:
            Dictionary with evaluation metrics
        """
        logger.info("Evaluating anomaly detection model")
        
        # Run anomaly detection
        anomaly_results = anomaly_detector.run_comprehensive_anomaly_detection(features_df)
        
        evaluation_metrics = {
            'model_type': 'anomaly_detection',
            'n_entities': len(anomaly_results) if not anomaly_results.empty else 0
        }
        
        if anomaly_results.empty:
            logger.warning("No anomaly results to evaluate")
            return evaluation_metrics
        
        # Count anomalies by type
        anomaly_counts = {}
        anomaly_cols = [col for col in anomaly_results.columns if 'anomaly' in col and col != 'overall_anomaly_flag']
        
        for col in anomaly_cols:
            if anomaly_results[col].dtype == bool:
                anomaly_counts[col] = anomaly_results[col].sum()
        
        evaluation_metrics['anomaly_counts'] = anomaly_counts
        
        # Overall anomaly statistics
        if 'overall_anomaly_count' in anomaly_results.columns:
            overall_stats = {
                'total_anomalies': (anomaly_results['overall_anomaly_count'] > 0).sum(),
                'high_anomaly_entities': (anomaly_results['overall_anomaly_count'] >= 3).sum(),
                'anomaly_distribution': anomaly_results['overall_anomaly_count'].value_counts().to_dict()
            }
            evaluation_metrics['overall_anomaly_stats'] = overall_stats
        
        # If known anomalies are provided for validation
        if known_anomalies and 'overall_anomaly_flag' in anomaly_results.columns:
            # Create ground truth labels
            y_true = anomaly_results.index.isin(known_anomalies)
            y_pred = anomaly_results['overall_anomaly_flag']
            
            # Calculate performance metrics
            evaluation_metrics['validation_metrics'] = {
                'precision': precision_score(y_true, y_pred),
                'recall': recall_score(y_true, y_pred),
                'f1_score': f1_score(y_true, y_pred),
                'accuracy': accuracy_score(y_true, y_pred)
            }
            
            # Confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            evaluation_metrics['confusion_matrix'] = {
                'tn': int(cm[0, 0]), 'fp': int(cm[0, 1]),
                'fn': int(cm[1, 0]), 'tp': int(cm[1, 1])
            }
        
        self.evaluation_results['anomaly_detection'] = evaluation_metrics
        return evaluation_metrics
    
    def cross_validate_models(self, 
                            quality_scorer,
                            features_df: pd.DataFrame,
                            cv_folds: int = 5) -> Dict:
        """
        Perform cross-validation on quality scoring models.
        
        Args:
            quality_scorer: Quality scoring model
            features_df: Feature DataFrame
            cv_folds: Number of cross-validation folds
            
        Returns:
            Cross-validation results
        """
        logger.info(f"Performing {cv_folds}-fold cross-validation")
        
        # Create quality labels
        quality_labels = quality_scorer.create_quality_labels(features_df)
        X = quality_scorer.prepare_features_for_scoring(quality_scorer.channel_features)
        
        # Only use labeled data
        labeled_mask = quality_labels.notna()
        X_labeled = X[labeled_mask]
        y_labeled = quality_labels[labeled_mask]
        
        if len(X_labeled) < cv_folds:
            logger.warning(f"Insufficient labeled data for {cv_folds}-fold CV")
            return {}
        
        cv_results = {}
        
        # Scale features
        from sklearn.preprocessing import RobustScaler
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X_labeled)
        
        # Cross-validation for different models
        models_to_test = {
            'gradient_boosting': quality_scorer.models.get('gradient_boosting'),
            'xgboost': quality_scorer.models.get('xgboost')
        }
        
        for model_name, model in models_to_test.items():
            if model is None:
                continue
                
            try:
                # Regression metrics
                mse_scores = cross_val_score(model, X_scaled, y_labeled, 
                                           cv=cv_folds, scoring='neg_mean_squared_error')
                mae_scores = cross_val_score(model, X_scaled, y_labeled, 
                                           cv=cv_folds, scoring='neg_mean_absolute_error')
                r2_scores = cross_val_score(model, X_scaled, y_labeled, 
                                          cv=cv_folds, scoring='r2')
                
                cv_results[model_name] = {
                    'mse_mean': -mse_scores.mean(),
                    'mse_std': mse_scores.std(),
                    'mae_mean': -mae_scores.mean(),
                    'mae_std': mae_scores.std(),
                    'r2_mean': r2_scores.mean(),
                    'r2_std': r2_scores.std()
                }
                
                logger.info(f"{model_name} CV - R2: {r2_scores.mean():.3f} ± {r2_scores.std():.3f}")
                
            except Exception as e:
                logger.error(f"Cross-validation failed for {model_name}: {e}")
                continue
        
        return cv_results
    
    def generate_evaluation_report(self) -> str:
        """
        Generate comprehensive evaluation report.
        
        Returns:
            Formatted evaluation report string
        """
        report_lines = ["=" * 60]
        report_lines.append("FRAUD DETECTION MODEL EVALUATION REPORT")
        report_lines.append("=" * 60)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        for model_type, metrics in self.evaluation_results.items():
            report_lines.append(f"\n{model_type.upper().replace('_', ' ')} MODEL")
            report_lines.append("-" * 40)
            
            if model_type == 'quality_scoring':
                report_lines.append(f"Channels Evaluated: {metrics['n_channels']:,}")
                
                if 'score_distribution' in metrics:
                    dist = metrics['score_distribution']
                    report_lines.append(f"Score Distribution:")
                    report_lines.append(f"  Mean: {dist['mean']:.2f}")
                    report_lines.append(f"  Std:  {dist['std']:.2f}")
                    report_lines.append(f"  Range: [{dist['min']:.1f}, {dist['max']:.1f}]")
                
                if 'category_distribution' in metrics:
                    report_lines.append(f"Quality Categories:")
                    for category, count in metrics['category_distribution'].items():
                        pct = count / metrics['n_channels'] * 100
                        report_lines.append(f"  {category}: {count:,} ({pct:.1f}%)")
                
                if 'regression_metrics' in metrics:
                    reg_metrics = metrics['regression_metrics']
                    report_lines.append(f"Validation Metrics:")
                    report_lines.append(f"  R²: {reg_metrics['r2']:.3f}")
                    report_lines.append(f"  MAE: {reg_metrics['mae']:.3f}")
                    report_lines.append(f"  Correlation: {reg_metrics['correlation']:.3f}")
            
            elif model_type == 'traffic_similarity':
                report_lines.append(f"Channels Analyzed: {metrics['n_channels']:,}")
                
                if 'clustering_metrics' in metrics:
                    cluster_metrics = metrics['clustering_metrics']
                    report_lines.append(f"Clustering Performance:")
                    report_lines.append(f"  Clusters: {cluster_metrics['n_clusters']}")
                    report_lines.append(f"  Silhouette Score: {cluster_metrics['silhouette_score']:.3f}")
                
                if 'outlier_detection' in metrics:
                    outlier_metrics = metrics['outlier_detection']
                    report_lines.append(f"Outlier Detection:")
                    report_lines.append(f"  Outliers: {outlier_metrics['n_outliers']} ({outlier_metrics['outlier_percentage']:.1f}%)")
            
            elif model_type == 'anomaly_detection':
                report_lines.append(f"Entities Analyzed: {metrics['n_entities']:,}")
                
                if 'anomaly_counts' in metrics:
                    report_lines.append(f"Anomalies Detected:")
                    for anomaly_type, count in metrics['anomaly_counts'].items():
                        report_lines.append(f"  {anomaly_type}: {count}")
                
                if 'validation_metrics' in metrics:
                    val_metrics = metrics['validation_metrics']
                    report_lines.append(f"Validation Performance:")
                    report_lines.append(f"  Precision: {val_metrics['precision']:.3f}")
                    report_lines.append(f"  Recall: {val_metrics['recall']:.3f}")
                    report_lines.append(f"  F1-Score: {val_metrics['f1_score']:.3f}")
        
        report_lines.append("\n" + "=" * 60)
        return "\n".join(report_lines)
    
    def monitor_model_drift(self, 
                          current_features: pd.DataFrame,
                          baseline_features: pd.DataFrame) -> Dict:
        """
        Monitor for model drift by comparing current vs baseline feature distributions.
        
        Args:
            current_features: Current feature data
            baseline_features: Baseline feature data for comparison
            
        Returns:
            Dictionary with drift detection results
        """
        logger.info("Monitoring model drift")
        
        drift_results = {
            'drift_detected': False,
            'drift_score': 0.0,
            'feature_drifts': {}
        }
        
        # Compare feature distributions
        common_features = current_features.columns.intersection(baseline_features.columns)
        
        for feature in common_features:
            if current_features[feature].dtype in ['int64', 'float64']:
                # KS test for numerical features
                from scipy.stats import ks_2samp
                
                # Remove NaN values
                current_vals = current_features[feature].dropna()
                baseline_vals = baseline_features[feature].dropna()
                
                if len(current_vals) > 0 and len(baseline_vals) > 0:
                    ks_stat, p_value = ks_2samp(current_vals, baseline_vals)
                    
                    drift_results['feature_drifts'][feature] = {
                        'ks_statistic': ks_stat,
                        'p_value': p_value,
                        'drift_detected': p_value < 0.05,
                        'current_mean': current_vals.mean(),
                        'baseline_mean': baseline_vals.mean(),
                        'mean_shift': abs(current_vals.mean() - baseline_vals.mean()) / baseline_vals.std()
                    }
        
        # Calculate overall drift score
        drift_features = sum(1 for drift_info in drift_results['feature_drifts'].values() 
                           if drift_info['drift_detected'])
        drift_results['drift_score'] = drift_features / len(common_features) if common_features else 0
        drift_results['drift_detected'] = drift_results['drift_score'] > 0.1  # 10% threshold
        
        return drift_results
    
    def save_evaluation_results(self, filepath: str):
        """Save evaluation results to file."""
        evaluation_data = {
            'evaluation_results': self.evaluation_results,
            'baseline_metrics': self.baseline_metrics,
            'model_history': self.model_history,
            'timestamp': datetime.now().isoformat()
        }
        joblib.dump(evaluation_data, filepath)
        logger.info(f"Evaluation results saved to {filepath}")
    
    def load_evaluation_results(self, filepath: str):
        """Load evaluation results from file."""
        evaluation_data = joblib.load(filepath)
        self.evaluation_results = evaluation_data['evaluation_results']
        self.baseline_metrics = evaluation_data['baseline_metrics']
        self.model_history = evaluation_data['model_history']
        logger.info(f"Evaluation results loaded from {filepath}")

def main():
    """Test model evaluation framework."""
    from data_pipeline import DataPipeline
    from feature_engineering import FeatureEngineer
    from quality_scoring import QualityScorer
    from traffic_similarity import TrafficSimilarityModel
    from anomaly_detection import AnomalyDetector
    
    # Load sample data
    pipeline = DataPipeline("/home/fiod/shimshi/bq-results-20250804-141411-1754316868932.csv")
    df = pipeline.load_data_chunked(sample_fraction=0.03)  # 3% sample for faster evaluation
    
    # Engineer features
    feature_engineer = FeatureEngineer()
    features_df = feature_engineer.create_all_features(df)
    channel_features = feature_engineer.create_channel_features(features_df)
    
    # Initialize evaluator
    evaluator = ModelEvaluator()
    
    # Train and evaluate quality scoring model
    quality_scorer = QualityScorer()
    quality_metrics = evaluator.evaluate_quality_scoring_model(quality_scorer, features_df)
    
    # Train and evaluate similarity model
    similarity_model = TrafficSimilarityModel(n_clusters=6)
    similarity_model.fit(channel_features)
    similarity_metrics = evaluator.evaluate_similarity_model(similarity_model, channel_features)
    
    # Train and evaluate anomaly detection model
    anomaly_detector = AnomalyDetector(contamination=0.1)
    anomaly_metrics = evaluator.evaluate_anomaly_detection_model(anomaly_detector, features_df)
    
    # Generate comprehensive report
    report = evaluator.generate_evaluation_report()
    logger.info("Evaluation Report:")
    logger.info(report)
    
    # Save evaluation results
    evaluator.save_evaluation_results("/home/fiod/shimshi/model_evaluation_results.pkl")
    
    return evaluator, quality_metrics, similarity_metrics, anomaly_metrics

if __name__ == "__main__":
    evaluator, quality_metrics, similarity_metrics, anomaly_metrics = main()