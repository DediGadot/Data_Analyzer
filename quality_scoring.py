"""
Quality Scoring Model
Creates a 1-10 scoring system for each channelId using supervised/semi-supervised approaches.
Combines fraud indicators with behavioral patterns to assess traffic quality.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor, IsolationForest
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
from sklearn.cluster import KMeans
import xgboost as xgb
import logging
from typing import Dict, List, Tuple, Optional
import joblib
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class QualityScorer:
    """
    Traffic quality scoring model that combines supervised and semi-supervised learning
    to generate 1-10 quality scores for each channel based on fraud indicators and behavior.
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.scaler = None
        self.models = {}
        self.feature_names = []
        self.quality_thresholds = {}
        
    def create_quality_labels(self, features_df: pd.DataFrame) -> pd.Series:
        """
        Create initial quality labels based on fraud indicators and behavioral patterns.
        This creates a semi-supervised dataset where we have labels for obvious cases.
        
        Args:
            features_df: DataFrame with engineered features
            
        Returns:
            Series with quality scores (1-10) where NaN indicates unlabeled data
        """
        logger.info("Creating initial quality labels based on fraud indicators")
        
        # Aggregate by channel
        channel_agg = features_df.groupby('channelId').agg({
            # Fraud indicators (lower is better)
            'isLikelyBot': 'mean',
            'total_fraud_flags': 'mean',
            'weighted_fraud_score': 'mean',
            
            # Behavioral indicators
            'ip': 'nunique',  # IP diversity
            'country': 'nunique',  # Geographic diversity  
            'browser': 'nunique',  # Browser diversity
            'ua_suspicious': 'mean',  # User agent suspicion
            'date': 'count',  # Volume
            
            # Temporal patterns
            'is_business_hours': 'mean',
            'is_weekend': 'mean',
            'hour': 'std',  # Time pattern diversity
            
            # Traffic patterns
            'keyword_commercial_intent': 'mean',
            'referrer_is_ad_network': 'mean'
        }).fillna(0)
        
        # Rename columns for clarity
        channel_agg.columns = [
            'bot_rate', 'fraud_flags_avg', 'fraud_score_avg',
            'ip_diversity', 'country_diversity', 'browser_diversity', 
            'ua_suspicious_rate', 'volume',
            'business_hours_rate', 'weekend_rate', 'time_pattern_diversity',
            'commercial_intent_rate', 'ad_network_rate'
        ]
        
        # Initialize quality scores as NaN (unlabeled)
        quality_scores = pd.Series(index=channel_agg.index, dtype=float)
        
        # Define clear quality rules for labeling
        
        # Very High Quality (Score 9-10): Low fraud, diverse traffic, business patterns
        high_quality_mask = (
            (channel_agg['bot_rate'] < 0.05) &
            (channel_agg['fraud_score_avg'] < 0.5) &
            (channel_agg['ip_diversity'] >= 10) &
            (channel_agg['country_diversity'] >= 2) &
            (channel_agg['business_hours_rate'] > 0.3) &
            (channel_agg['volume'] >= 50)
        )
        quality_scores[high_quality_mask] = np.random.uniform(8.5, 10.0, high_quality_mask.sum())
        
        # High Quality (Score 7-8): Moderate fraud, good diversity
        good_quality_mask = (
            (channel_agg['bot_rate'] < 0.15) &
            (channel_agg['fraud_score_avg'] < 1.0) &
            (channel_agg['ip_diversity'] >= 5) &
            (channel_agg['volume'] >= 20) &
            ~high_quality_mask
        )
        quality_scores[good_quality_mask] = np.random.uniform(7.0, 8.5, good_quality_mask.sum())
        
        # Very Low Quality (Score 1-3): High fraud, suspicious patterns
        low_quality_mask = (
            (channel_agg['bot_rate'] > 0.5) |
            (channel_agg['fraud_score_avg'] > 3.0) |
            (channel_agg['ua_suspicious_rate'] > 0.3) |
            ((channel_agg['ip_diversity'] == 1) & (channel_agg['volume'] > 100))  # Single IP, high volume
        )
        quality_scores[low_quality_mask] = np.random.uniform(1.0, 3.0, low_quality_mask.sum())
        
        # Low Quality (Score 3-5): Moderate fraud or suspicious patterns
        medium_low_quality_mask = (
            ((channel_agg['bot_rate'] > 0.25) & (channel_agg['bot_rate'] <= 0.5)) |
            ((channel_agg['fraud_score_avg'] > 1.5) & (channel_agg['fraud_score_avg'] <= 3.0)) |
            (channel_agg['ip_diversity'] < 3) |
            (channel_agg['time_pattern_diversity'] < 1.0)  # Very narrow time patterns
        ) & ~low_quality_mask
        quality_scores[medium_low_quality_mask] = np.random.uniform(3.0, 5.0, medium_low_quality_mask.sum())
        
        # Medium Quality (Score 5-7): Average patterns
        medium_quality_mask = (
            (channel_agg['bot_rate'] <= 0.25) &
            (channel_agg['fraud_score_avg'] <= 1.5) &
            (channel_agg['ip_diversity'] >= 3) &
            ~high_quality_mask & ~good_quality_mask & ~low_quality_mask & ~medium_low_quality_mask
        )
        quality_scores[medium_quality_mask] = np.random.uniform(5.0, 7.0, medium_quality_mask.sum())
        
        # Store the aggregated features for model training
        self.channel_features = channel_agg
        
        labeled_count = quality_scores.notna().sum()
        total_count = len(quality_scores)
        logger.info(f"Created labels for {labeled_count}/{total_count} channels ({labeled_count/total_count:.1%})")
        
        return quality_scores
    
    def prepare_features_for_scoring(self, channel_features: pd.DataFrame) -> pd.DataFrame:
        """Prepare channel features for quality scoring."""
        
        # Select relevant features for quality prediction
        feature_cols = [
            'bot_rate', 'fraud_flags_avg', 'fraud_score_avg',
            'ip_diversity', 'country_diversity', 'browser_diversity',
            'ua_suspicious_rate', 'volume',
            'business_hours_rate', 'weekend_rate', 'time_pattern_diversity',
            'commercial_intent_rate', 'ad_network_rate'
        ]
        
        # Handle missing features
        available_cols = [col for col in feature_cols if col in channel_features.columns]
        X = channel_features[available_cols].fillna(0)
        
        # Add derived features
        if 'volume' in X.columns and 'ip_diversity' in X.columns:
            X['requests_per_ip'] = X['volume'] / (X['ip_diversity'] + 1)
        
        if 'bot_rate' in X.columns and 'fraud_score_avg' in X.columns:
            X['combined_fraud_risk'] = X['bot_rate'] * 0.6 + X['fraud_score_avg'] * 0.4
        
        # Log transform volume to handle skewness
        if 'volume' in X.columns:
            X['log_volume'] = np.log1p(X['volume'])
        
        return X
    
    def fit_supervised_model(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """
        Fit supervised models on labeled data.
        
        Args:
            X: Feature matrix
            y: Quality scores
            
        Returns:
            Dictionary with model results
        """
        logger.info("Fitting supervised quality scoring models")
        
        # Only use labeled data
        labeled_mask = y.notna()
        X_labeled = X[labeled_mask]
        y_labeled = y[labeled_mask]
        
        if len(X_labeled) < 10:
            logger.warning("Insufficient labeled data for supervised learning")
            return {}
        
        # Scale features
        self.scaler = RobustScaler()
        X_scaled = self.scaler.fit_transform(X_labeled)
        self.feature_names = X.columns.tolist()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_labeled, test_size=0.2, random_state=self.random_state
        )
        
        results = {}
        
        # 1. Gradient Boosting Regressor
        logger.info("Training Gradient Boosting model")
        gbr = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=self.random_state
        )
        gbr.fit(X_train, y_train)
        gbr_pred = gbr.predict(X_test)
        
        self.models['gradient_boosting'] = gbr
        results['gradient_boosting'] = {
            'mse': mean_squared_error(y_test, gbr_pred),
            'mae': mean_absolute_error(y_test, gbr_pred),
            'r2': r2_score(y_test, gbr_pred),
            'feature_importance': dict(zip(self.feature_names, gbr.feature_importances_))
        }
        
        # 2. XGBoost Regressor
        logger.info("Training XGBoost model")
        xgb_model = xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=self.random_state,
            eval_metric='rmse'
        )
        xgb_model.fit(X_train, y_train)
        xgb_pred = xgb_model.predict(X_test)
        
        self.models['xgboost'] = xgb_model
        results['xgboost'] = {
            'mse': mean_squared_error(y_test, xgb_pred),
            'mae': mean_absolute_error(y_test, xgb_pred),
            'r2': r2_score(y_test, xgb_pred),
            'feature_importance': dict(zip(self.feature_names, xgb_model.feature_importances_))
        }
        
        # 3. Random Forest (for comparison)
        logger.info("Training Random Forest model")
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=self.random_state
        )
        
        # Convert continuous scores to discrete classes for classification
        y_classes = pd.cut(y_labeled, bins=[0, 3, 5, 7, 10], labels=['Low', 'Medium-Low', 'Medium-High', 'High'])
        y_train_class = pd.cut(y_train, bins=[0, 3, 5, 7, 10], labels=['Low', 'Medium-Low', 'Medium-High', 'High'])
        y_test_class = pd.cut(y_test, bins=[0, 3, 5, 7, 10], labels=['Low', 'Medium-Low', 'Medium-High', 'High'])
        
        rf.fit(X_train, y_train_class)
        rf_pred_proba = rf.predict_proba(X_test)
        
        self.models['random_forest'] = rf
        self.quality_thresholds = {'Low': 2, 'Medium-Low': 4, 'Medium-High': 6, 'High': 8}
        
        return results
    
    def fit_semi_supervised_model(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """
        Fit semi-supervised models to propagate labels to unlabeled channels.
        
        Args:
            X: Feature matrix
            y: Quality scores (with NaN for unlabeled data)
            
        Returns:
            Dictionary with model results
        """
        logger.info("Fitting semi-supervised quality scoring models")
        
        # Prepare data for semi-supervised learning
        self.scaler = RobustScaler()
        X_scaled = self.scaler.fit_transform(X)
        self.feature_names = X.columns.tolist()
        
        # Convert continuous scores to discrete classes for semi-supervised learning
        # Convert quality scores to discrete classes (1-4 representing quality levels)
        y_discrete = pd.cut(y, bins=[0, 2.5, 5, 7.5, 10], labels=[1, 2, 3, 4], include_lowest=True)
        # Convert categorical to numeric, then handle NaN for semi-supervised algorithms
        y_numeric = pd.to_numeric(y_discrete, errors='coerce')
        # Convert NaN to -1 for semi-supervised algorithms (unlabeled)
        y_semi = y_numeric.fillna(-1).astype(int)
        
        results = {}
        
        # 1. Label Propagation
        logger.info("Training Label Propagation model")
        lp = LabelPropagation(kernel='knn', n_neighbors=7, max_iter=1000)
        lp.fit(X_scaled, y_semi)
        
        self.models['label_propagation'] = lp
        
        # Predict on all data
        lp_pred_discrete = lp.predict(X_scaled)
        
        # Convert discrete classes back to continuous scores
        # Map classes 1,2,3,4 to score ranges 1.25, 3.75, 6.25, 8.75
        class_to_score = {1: 1.25, 2: 3.75, 3: 6.25, 4: 8.75, -1: 5.0}  # -1 for unlabeled gets median
        lp_pred = np.array([class_to_score.get(int(cls), 5.0) for cls in lp_pred_discrete])
        
        labeled_mask = y.notna()
        
        if labeled_mask.sum() > 0:
            # Evaluate on labeled data
            lp_pred_labeled = lp_pred[labeled_mask]
            y_true = y[labeled_mask]
            
            results['label_propagation'] = {
                'mse': mean_squared_error(y_true, lp_pred_labeled),
                'mae': mean_absolute_error(y_true, lp_pred_labeled),
                'r2': r2_score(y_true, lp_pred_labeled),
                'predictions': lp_pred
            }
        
        # 2. Label Spreading
        logger.info("Training Label Spreading model")
        ls = LabelSpreading(kernel='knn', n_neighbors=7, max_iter=1000, alpha=0.8)
        ls.fit(X_scaled, y_semi)
        
        self.models['label_spreading'] = ls
        
        # Predict on all data
        ls_pred = ls.predict(X_scaled)
        
        if labeled_mask.sum() > 0:
            ls_pred_labeled = ls_pred[labeled_mask]
            
            results['label_spreading'] = {
                'mse': mean_squared_error(y_true, ls_pred_labeled),
                'mae': mean_absolute_error(y_true, ls_pred_labeled),
                'r2': r2_score(y_true, ls_pred_labeled),
                'predictions': ls_pred
            }
        
        return results
    
    def ensemble_predictions(self, X: pd.DataFrame, methods: List[str] = None) -> pd.Series:
        """
        Create ensemble predictions from multiple models.
        
        Args:
            X: Feature matrix
            methods: List of model names to ensemble
            
        Returns:
            Series with ensemble quality scores
        """
        if methods is None:
            methods = ['gradient_boosting', 'xgboost', 'label_propagation']
        
        available_methods = [m for m in methods if m in self.models]
        
        if not available_methods:
            logger.warning("No trained models available for ensemble")
            return pd.Series(index=X.index, dtype=float)
        
        X_scaled = self.scaler.transform(X)
        predictions = []
        weights = []
        
        for method in available_methods:
            model = self.models[method]
            
            if method in ['gradient_boosting', 'xgboost']:
                pred = model.predict(X_scaled)
                weight = 0.4  # Higher weight for supervised models
            elif method in ['label_propagation', 'label_spreading']:
                pred = model.predict(X_scaled)
                weight = 0.3  # Lower weight for semi-supervised models
            else:
                continue
            
            # Ensure predictions are in valid range [1, 10]
            pred = np.clip(pred, 1, 10)
            predictions.append(pred)
            weights.append(weight)
        
        if not predictions:
            return pd.Series(index=X.index, dtype=float)
        
        # Weighted average
        weights = np.array(weights) / sum(weights)  # Normalize weights
        ensemble_pred = np.average(predictions, axis=0, weights=weights)
        
        return pd.Series(ensemble_pred, index=X.index)
    
    def score_channels(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Score all channels and return detailed results.
        
        Args:
            features_df: DataFrame with traffic features
            
        Returns:
            DataFrame with channel scores and explanations
        """
        logger.info("Scoring all channels")
        
        # Create quality labels
        quality_labels = self.create_quality_labels(features_df)
        
        # Prepare features
        X = self.prepare_features_for_scoring(self.channel_features)
        
        # Fit models
        supervised_results = self.fit_supervised_model(X, quality_labels)
        semi_supervised_results = self.fit_semi_supervised_model(X, quality_labels)
        
        # Generate ensemble predictions
        ensemble_scores = self.ensemble_predictions(X)
        
        # Create results DataFrame
        results_df = pd.DataFrame({
            'channelId': X.index,
            'quality_score': ensemble_scores,
            'original_label': quality_labels,
            'volume': self.channel_features['volume'],
            'bot_rate': self.channel_features['bot_rate'],
            'fraud_score_avg': self.channel_features['fraud_score_avg'],
            'ip_diversity': self.channel_features['ip_diversity'],
            'country_diversity': self.channel_features['country_diversity']
        }).set_index('channelId')
        
        # Add quality categories
        results_df['quality_category'] = pd.cut(
            results_df['quality_score'],
            bins=[0, 3, 5, 7, 10],
            labels=['Low', 'Medium-Low', 'Medium-High', 'High']
        )
        
        # Add risk flags
        results_df['high_risk'] = (
            (results_df['bot_rate'] > 0.3) |
            (results_df['fraud_score_avg'] > 2.0) |
            (results_df['quality_score'] < 3.0)
        )
        
        # Sort by quality score
        results_df = results_df.sort_values('quality_score', ascending=False)
        
        logger.info(f"Scored {len(results_df)} channels")
        logger.info(f"Quality distribution: {results_df['quality_category'].value_counts().to_dict()}")
        
        return results_df
    
    def get_feature_importance(self) -> Dict:
        """Get feature importance from trained models."""
        importance_summary = {}
        
        for model_name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                importance_summary[model_name] = dict(zip(
                    self.feature_names, 
                    model.feature_importances_
                ))
        
        return importance_summary
    
    def save_model(self, filepath: str):
        """Save the trained model."""
        model_data = {
            'scaler': self.scaler,
            'models': self.models,
            'feature_names': self.feature_names,
            'quality_thresholds': self.quality_thresholds,
            'channel_features': self.channel_features
        }
        joblib.dump(model_data, filepath)
        logger.info(f"Quality scoring model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model."""
        model_data = joblib.load(filepath)
        self.scaler = model_data['scaler']
        self.models = model_data['models']
        self.feature_names = model_data['feature_names']
        self.quality_thresholds = model_data['quality_thresholds']
        self.channel_features = model_data['channel_features']
        logger.info(f"Quality scoring model loaded from {filepath}")

def main():
    """Test quality scoring model."""
    from data_pipeline import DataPipeline
    from feature_engineering import FeatureEngineer
    
    # Load sample data
    pipeline = DataPipeline("/home/fiod/shimshi/bq-results-20250804-141411-1754316868932.csv")
    df = pipeline.load_data_chunked(sample_fraction=0.05)  # 5% sample
    
    # Engineer features
    feature_engineer = FeatureEngineer()
    features_df = feature_engineer.create_all_features(df)
    
    # Score channels
    quality_scorer = QualityScorer()
    results_df = quality_scorer.score_channels(features_df)
    
    # Display results
    logger.info("Top 10 highest quality channels:")
    logger.info(results_df.head(10)[['quality_score', 'quality_category', 'volume', 'bot_rate']])
    
    logger.info("Bottom 10 lowest quality channels:")
    logger.info(results_df.tail(10)[['quality_score', 'quality_category', 'volume', 'bot_rate']])
    
    # Feature importance
    importance = quality_scorer.get_feature_importance()
    logger.info(f"Feature importance: {importance}")
    
    # Save model
    quality_scorer.save_model("/home/fiod/shimshi/quality_scoring_model.pkl")
    
    return quality_scorer, results_df

if __name__ == "__main__":
    quality_scorer, results_df = main()