"""
Pattern Anomaly Detection Models
Detects suspicious temporal, device, geographic, and behavioral patterns 
that individual requests might miss using advanced anomaly detection techniques.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from scipy import stats
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional, Any
import joblib
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class AnomalyDetector:
    """
    Multi-layered anomaly detection system for identifying suspicious patterns
    in advertising traffic that might indicate fraud or bot activity.
    """
    
    def __init__(self, contamination: float = 0.1, random_state: int = 42):
        self.contamination = contamination
        self.random_state = random_state
        self.scalers = {}
        self.models = {}
        self.pattern_baselines = {}
        
    def detect_temporal_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect temporal anomalies in traffic patterns.
        
        Args:
            df: DataFrame with traffic data including timestamps
            
        Returns:
            DataFrame with temporal anomaly scores and flags
        """
        logger.info("Detecting temporal anomalies")
        
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        
        # Create time-based aggregations
        temporal_features = self._create_temporal_aggregations(df)
        
        # Detect anomalies in different temporal patterns
        anomaly_results = {}
        
        # 1. Hourly traffic pattern anomalies
        hourly_anomalies = self._detect_hourly_anomalies(temporal_features)
        anomaly_results['hourly'] = hourly_anomalies
        
        # 2. Daily pattern anomalies
        daily_anomalies = self._detect_daily_anomalies(temporal_features)
        anomaly_results['daily'] = daily_anomalies
        
        # 3. Burst detection (unusual spikes in short time windows)
        burst_anomalies = self._detect_traffic_bursts(df)
        anomaly_results['burst'] = burst_anomalies
        
        # Combine temporal anomaly scores
        combined_scores = self._combine_anomaly_scores(anomaly_results)
        
        return combined_scores
    
    def detect_geographic_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect geographic pattern anomalies.
        
        Args:
            df: DataFrame with traffic data including country, IP
            
        Returns:
            DataFrame with geographic anomaly scores
        """
        logger.info("Detecting geographic anomalies")
        
        if 'country' not in df.columns:
            logger.warning("No country data available for geographic anomaly detection")
            return pd.DataFrame()
        
        # Country distribution anomalies
        country_patterns = df.groupby(['channelId', 'country']).size().unstack(fill_value=0)
        
        # Detect channels with unusual country distributions
        scaler = StandardScaler()
        country_scaled = scaler.fit_transform(country_patterns)
        
        # Use Isolation Forest to detect unusual geographic patterns
        iso_forest = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state
        )
        geo_anomaly_scores = iso_forest.fit_predict(country_scaled)
        geo_anomaly_scores = iso_forest.decision_function(country_scaled)
        
        geo_results = pd.DataFrame({
            'channelId': country_patterns.index,
            'geo_anomaly_score': geo_anomaly_scores,
            'geo_is_anomaly': geo_anomaly_scores < 0,
            'country_diversity': country_patterns.gt(0).sum(axis=1),
            'dominant_country_pct': country_patterns.max(axis=1) / country_patterns.sum(axis=1)
        })
        
        # IP-based geographic anomalies
        if 'ip' in df.columns:
            ip_geo_anomalies = self._detect_ip_geographic_anomalies(df)
            geo_results = geo_results.merge(ip_geo_anomalies, on='channelId', how='left')
        
        self.models['geographic'] = {
            'isolation_forest': iso_forest,
            'scaler': scaler,
            'country_patterns': country_patterns
        }
        
        return geo_results
    
    def detect_device_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect device and browser pattern anomalies.
        
        Args:
            df: DataFrame with device, browser, user agent data
            
        Returns:
            DataFrame with device anomaly scores
        """
        logger.info("Detecting device pattern anomalies")
        
        device_features = []
        
        # Device/Browser combination patterns
        if 'device' in df.columns and 'browser' in df.columns:
            device_combo = df.groupby(['channelId', 'device', 'browser']).size().unstack(fill_value=0)
            device_features.append(device_combo)
        
        # User agent pattern analysis
        if 'userAgent' in df.columns:
            ua_features = self._analyze_user_agent_patterns(df)
            device_features.append(ua_features)
        
        # Browser version anomalies
        if 'browserMajorVersion' in df.columns:
            version_features = self._analyze_browser_version_patterns(df)
            device_features.append(version_features)
        
        if not device_features:
            logger.warning("No device data available for anomaly detection")
            return pd.DataFrame()
        
        # Combine all device features
        combined_features = pd.concat(device_features, axis=1)
        combined_features = combined_features.fillna(0)
        
        # Detect anomalies
        scaler = RobustScaler()
        features_scaled = scaler.fit_transform(combined_features)
        
        # Multiple anomaly detection methods
        detectors = {
            'isolation_forest': IsolationForest(contamination=self.contamination, random_state=self.random_state),
            'lof': LocalOutlierFactor(contamination=self.contamination),
            'one_class_svm': OneClassSVM(nu=self.contamination)
        }
        
        device_results = pd.DataFrame(index=combined_features.index)
        
        for name, detector in detectors.items():
            if name == 'lof':
                scores = detector.fit_predict(features_scaled)
                decision_scores = detector.negative_outlier_factor_
            else:
                scores = detector.fit_predict(features_scaled)
                decision_scores = detector.decision_function(features_scaled)
            
            device_results[f'device_{name}_anomaly'] = scores == -1
            device_results[f'device_{name}_score'] = decision_scores
        
        # Ensemble anomaly score
        device_results['device_anomaly_ensemble'] = (
            device_results[[col for col in device_results.columns if 'anomaly' in col and 'ensemble' not in col]].sum(axis=1) >= 2
        )
        
        self.models['device'] = {
            'detectors': detectors,
            'scaler': scaler,
            'features': combined_features.columns.tolist()
        }
        
        return device_results
    
    def detect_behavioral_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect behavioral pattern anomalies (request patterns, sequences, etc.).
        
        Args:
            df: DataFrame with behavioral features
            
        Returns:
            DataFrame with behavioral anomaly scores
        """
        logger.info("Detecting behavioral anomalies")
        
        behavioral_features = self._create_behavioral_features(df)
        
        if behavioral_features.empty:
            logger.warning("No behavioral features available")
            return pd.DataFrame()
        
        # Scale features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(behavioral_features)
        
        # Advanced anomaly detection using multiple methods
        anomaly_methods = {
            'isolation_forest': IsolationForest(
                contamination=self.contamination,
                random_state=self.random_state,
                n_estimators=100
            ),
            'elliptic_envelope': EllipticEnvelope(
                contamination=self.contamination,
                random_state=self.random_state
            ),
            'one_class_svm': OneClassSVM(nu=self.contamination, kernel='rbf')
        }
        
        behavioral_results = pd.DataFrame(index=behavioral_features.index)
        
        for name, detector in anomaly_methods.items():
            try:
                anomaly_labels = detector.fit_predict(features_scaled)
                if hasattr(detector, 'decision_function'):
                    anomaly_scores = detector.decision_function(features_scaled)
                else:
                    anomaly_scores = detector.score_samples(features_scaled)
                
                behavioral_results[f'behavioral_{name}_anomaly'] = anomaly_labels == -1
                behavioral_results[f'behavioral_{name}_score'] = anomaly_scores
                
            except Exception as e:
                logger.warning(f"Failed to fit {name}: {e}")
                continue
        
        # Create ensemble behavioral anomaly score
        anomaly_cols = [col for col in behavioral_results.columns if 'anomaly' in col and 'ensemble' not in col]
        if anomaly_cols:
            behavioral_results['behavioral_anomaly_ensemble'] = (
                behavioral_results[anomaly_cols].sum(axis=1) >= len(anomaly_cols) // 2
            )
        
        self.models['behavioral'] = {
            'methods': anomaly_methods,
            'scaler': scaler,
            'features': behavioral_features.columns.tolist()
        }
        
        return behavioral_results
    
    def detect_volume_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect volume-based anomalies (unusual traffic volumes).
        
        Args:
            df: DataFrame with traffic data
            
        Returns:
            DataFrame with volume anomaly information
        """
        logger.info("Detecting volume anomalies")
        
        # Aggregate by different time windows and entities
        volume_features = {}
        
        # Channel-level volume patterns
        if 'channelId' in df.columns:
            channel_volumes = df.groupby('channelId').agg({
                'date': 'count',
                'ip': 'nunique',
                'userId': 'nunique' if 'userId' in df.columns else 'count'
            })
            channel_volumes.columns = ['total_requests', 'unique_ips', 'unique_users']
            volume_features['channel'] = channel_volumes
        
        # IP-level volume patterns
        if 'ip' in df.columns:
            ip_volumes = df.groupby('ip').agg({
                'date': 'count',
                'channelId': 'nunique'
            })
            ip_volumes.columns = ['requests_per_ip', 'channels_per_ip']
            
            # Detect high-volume IPs
            volume_threshold = ip_volumes['requests_per_ip'].quantile(0.95)
            high_volume_ips = ip_volumes[ip_volumes['requests_per_ip'] > volume_threshold]
            
            volume_features['ip'] = {
                'high_volume_ips': high_volume_ips.index.tolist(),
                'volume_threshold': volume_threshold,
                'stats': ip_volumes.describe()
            }
        
        # Time-based volume anomalies
        df['hour'] = pd.to_datetime(df['date']).dt.hour
        hourly_volumes = df.groupby('hour').size()
        
        # Z-score based anomaly detection for hourly volumes
        z_scores = np.abs(stats.zscore(hourly_volumes))
        anomalous_hours = hourly_volumes[z_scores > 2].index.tolist()
        
        volume_features['temporal'] = {
            'anomalous_hours': anomalous_hours,
            'hourly_volumes': hourly_volumes.to_dict()
        }
        
        # Convert to DataFrame format
        volume_results = pd.DataFrame()
        
        if 'channel' in volume_features:
            channel_vol = volume_features['channel']
            
            # Detect volume anomalies using statistical methods
            for col in ['total_requests', 'unique_ips', 'unique_users']:
                if col in channel_vol.columns:
                    z_scores = np.abs(stats.zscore(channel_vol[col]))
                    channel_vol[f'{col}_z_score'] = z_scores
                    channel_vol[f'{col}_anomaly'] = z_scores > 2.5
            
            volume_results = channel_vol
        
        self.pattern_baselines['volume'] = volume_features
        
        return volume_results
    
    def _create_temporal_aggregations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create temporal aggregations for anomaly detection."""
        df['hour'] = df['date'].dt.hour
        df['day_of_week'] = df['date'].dt.dayofweek
        df['date_only'] = df['date'].dt.date
        
        # Hourly patterns by channel
        hourly_patterns = df.groupby(['channelId', 'hour']).size().unstack(fill_value=0)
        
        # Daily patterns by channel
        daily_patterns = df.groupby(['channelId', 'day_of_week']).size().unstack(fill_value=0)
        
        # Combine patterns
        temporal_features = pd.concat([hourly_patterns, daily_patterns], axis=1)
        temporal_features.columns = [f'hour_{col}' if col in range(24) else f'dow_{col}' 
                                   for col in temporal_features.columns]
        
        return temporal_features.fillna(0)
    
    def _detect_hourly_anomalies(self, temporal_features: pd.DataFrame) -> pd.DataFrame:
        """Detect anomalies in hourly traffic patterns."""
        hour_cols = [col for col in temporal_features.columns if col.startswith('hour_')]
        if not hour_cols:
            return pd.DataFrame()
        
        hourly_data = temporal_features[hour_cols]
        
        # Normalize by total traffic to get patterns
        hourly_patterns = hourly_data.div(hourly_data.sum(axis=1), axis=0).fillna(0)
        
        # Detect anomalies using Isolation Forest
        iso_forest = IsolationForest(contamination=self.contamination, random_state=self.random_state)
        anomaly_scores = iso_forest.fit_predict(hourly_patterns)
        decision_scores = iso_forest.decision_function(hourly_patterns)
        
        results = pd.DataFrame({
            'channelId': hourly_patterns.index,
            'hourly_anomaly': anomaly_scores == -1,
            'hourly_anomaly_score': decision_scores
        })
        
        return results
    
    def _detect_daily_anomalies(self, temporal_features: pd.DataFrame) -> pd.DataFrame:
        """Detect anomalies in daily traffic patterns."""
        dow_cols = [col for col in temporal_features.columns if col.startswith('dow_')]
        if not dow_cols:
            return pd.DataFrame()
        
        daily_data = temporal_features[dow_cols]
        daily_patterns = daily_data.div(daily_data.sum(axis=1), axis=0).fillna(0)
        
        # Statistical anomaly detection
        z_scores = np.abs(stats.zscore(daily_patterns, axis=0))
        daily_anomalies = (z_scores > 2).any(axis=1)
        
        results = pd.DataFrame({
            'channelId': daily_patterns.index,
            'daily_anomaly': daily_anomalies,
            'daily_max_z_score': z_scores.max(axis=1)
        })
        
        return results
    
    def _detect_traffic_bursts(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect unusual traffic bursts in short time windows."""
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['date'])
        
        # 5-minute window burst detection
        df['time_5min'] = df['timestamp'].dt.floor('5min')
        
        burst_detection = []
        
        for entity_col in ['channelId', 'ip']:
            if entity_col not in df.columns:
                continue
                
            # Count requests per entity per 5-minute window
            entity_time_counts = df.groupby([entity_col, 'time_5min']).size().reset_index(name='requests')
            
            # Calculate baseline (median) and burst threshold (95th percentile)
            entity_baselines = entity_time_counts.groupby(entity_col)['requests'].agg(['median', lambda x: x.quantile(0.95)])
            entity_baselines.columns = ['baseline', 'burst_threshold']
            
            # Identify bursts
            entity_time_counts = entity_time_counts.merge(entity_baselines, left_on=entity_col, right_index=True)
            entity_time_counts['is_burst'] = entity_time_counts['requests'] > entity_time_counts['burst_threshold']
            
            # Aggregate burst information per entity
            burst_summary = entity_time_counts.groupby(entity_col).agg({
                'is_burst': ['sum', 'mean'],
                'requests': 'max'
            })
            burst_summary.columns = [f'{entity_col}_burst_count', f'{entity_col}_burst_rate', f'{entity_col}_max_5min']
            
            burst_detection.append(burst_summary)
        
        if burst_detection:
            return pd.concat(burst_detection, axis=1)
        else:
            return pd.DataFrame()
    
    def _detect_ip_geographic_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect IP-based geographic anomalies."""
        # IPs appearing in multiple countries (potential proxy/VPN indicators)
        ip_countries = df.groupby('ip')['country'].nunique()
        multi_country_ips = ip_countries[ip_countries > 1]
        
        # Channel-level IP geographic diversity anomalies
        channel_ip_geo = df.groupby('channelId').agg({
            'ip': lambda x: (df[df['channelId'] == x.name]['ip'].map(ip_countries) > 1).sum(),
            'country': 'nunique'
        })
        channel_ip_geo.columns = ['multi_country_ips', 'country_diversity']
        
        # Flag channels with unusually high multi-country IP usage
        if len(channel_ip_geo) > 0:
            threshold = channel_ip_geo['multi_country_ips'].quantile(0.9)
            channel_ip_geo['ip_geo_anomaly'] = channel_ip_geo['multi_country_ips'] > threshold
        else:
            channel_ip_geo['ip_geo_anomaly'] = False
        
        return channel_ip_geo.reset_index()
    
    def _analyze_user_agent_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Analyze user agent patterns for anomalies."""
        if 'userAgent' not in df.columns:
            return pd.DataFrame()
        
        # User agent diversity and patterns by channel
        ua_analysis = df.groupby('channelId')['userAgent'].agg([
            'nunique',
            lambda x: x.str.len().mean(),
            lambda x: x.str.contains('bot|crawler|spider', case=False, na=False).mean()
        ])
        ua_analysis.columns = ['ua_diversity', 'ua_avg_length', 'ua_bot_rate']
        
        return ua_analysis
    
    def _analyze_browser_version_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Analyze browser version patterns."""
        if 'browserMajorVersion' not in df.columns:
            return pd.DataFrame()
        
        version_analysis = df.groupby('channelId')['browserMajorVersion'].agg([
            'nunique',
            'mean',
            'std',
            lambda x: (x < 70).mean(),  # Very old versions
            lambda x: (x > 120).mean()  # Very new versions
        ])
        version_analysis.columns = ['version_diversity', 'version_mean', 'version_std', 'old_version_rate', 'new_version_rate']
        
        return version_analysis.fillna(0)
    
    def _create_behavioral_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create behavioral features for anomaly detection."""
        behavioral_features = []
        
        # Request timing patterns
        if 'date' in df.columns:
            df_sorted = df.sort_values(['channelId', 'date'])
            timing_features = df_sorted.groupby('channelId').agg({
                'date': [
                    lambda x: x.diff().dt.total_seconds().mean(),  # Average time between requests
                    lambda x: x.diff().dt.total_seconds().std(),   # Std of time between requests
                    'count'  # Total requests
                ]
            })
            timing_features.columns = ['avg_request_interval', 'request_interval_std', 'total_requests']
            behavioral_features.append(timing_features)
        
        # Keyword diversity and patterns
        if 'keyword' in df.columns:
            keyword_features = df.groupby('channelId')['keyword'].agg([
                'nunique',
                lambda x: x.str.len().mean(),
                lambda x: x.str.split().str.len().mean()
            ])
            keyword_features.columns = ['keyword_diversity', 'keyword_avg_length', 'keyword_avg_words']
            behavioral_features.append(keyword_features)
        
        # Referrer patterns
        if 'referrer' in df.columns:
            referrer_features = df.groupby('channelId')['referrer'].agg([
                'nunique',
                lambda x: x.isna().mean()
            ])
            referrer_features.columns = ['referrer_diversity', 'referrer_missing_rate']
            behavioral_features.append(referrer_features)
        
        if behavioral_features:
            return pd.concat(behavioral_features, axis=1).fillna(0)
        else:
            return pd.DataFrame()
    
    def _combine_anomaly_scores(self, anomaly_results: Dict) -> pd.DataFrame:
        """Combine multiple anomaly detection results."""
        combined_results = pd.DataFrame()
        
        for anomaly_type, results in anomaly_results.items():
            if isinstance(results, pd.DataFrame) and not results.empty:
                if combined_results.empty:
                    combined_results = results.copy()
                else:
                    combined_results = combined_results.merge(results, on='channelId', how='outer')
        
        return combined_results.fillna(False)
    
    def run_comprehensive_detection(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run comprehensive anomaly detection across all pattern types.
        
        Args:
            df: Input DataFrame with traffic data
            
        Returns:
            DataFrame with comprehensive anomaly scores and flags
        """
        logger.info("Running comprehensive anomaly detection")
        
        results = {}
        
        # Temporal anomalies
        try:
            temporal_results = self.detect_temporal_anomalies(df)
            if not temporal_results.empty:
                results['temporal'] = temporal_results
        except Exception as e:
            logger.error(f"Temporal anomaly detection failed: {e}")
        
        # Geographic anomalies
        try:
            geo_results = self.detect_geographic_anomalies(df)
            if not geo_results.empty:
                results['geographic'] = geo_results
        except Exception as e:
            logger.error(f"Geographic anomaly detection failed: {e}")
        
        # Device anomalies
        try:
            device_results = self.detect_device_anomalies(df)
            if not device_results.empty:
                results['device'] = device_results
        except Exception as e:
            logger.error(f"Device anomaly detection failed: {e}")
        
        # Behavioral anomalies
        try:
            behavioral_results = self.detect_behavioral_anomalies(df)
            if not behavioral_results.empty:
                results['behavioral'] = behavioral_results
        except Exception as e:
            logger.error(f"Behavioral anomaly detection failed: {e}")
        
        # Volume anomalies
        try:
            volume_results = self.detect_volume_anomalies(df)
            if not volume_results.empty:
                results['volume'] = volume_results
        except Exception as e:
            logger.error(f"Volume anomaly detection failed: {e}")
        
        # Combine all results
        if not results:
            logger.warning("No anomaly detection results generated")
            return pd.DataFrame()
        
        # Merge all results
        final_results = None
        for anomaly_type, result_df in results.items():
            if final_results is None:
                final_results = result_df.copy()
            else:
                # Merge on common identifier (channelId or index)
                if 'channelId' in result_df.columns and 'channelId' in final_results.columns:
                    final_results = final_results.merge(result_df, on='channelId', how='outer')
                else:
                    final_results = final_results.merge(result_df, left_index=True, right_index=True, how='outer')
        
        # Create overall anomaly score
        if final_results is not None:
            anomaly_cols = [col for col in final_results.columns if 'anomaly' in col and 'overall' not in col]
            if anomaly_cols:
                final_results['overall_anomaly_count'] = final_results[anomaly_cols].sum(axis=1)
                final_results['overall_anomaly_flag'] = final_results['overall_anomaly_count'] >= 2
        
        logger.info(f"Comprehensive anomaly detection complete. Analyzed {len(final_results) if final_results is not None else 0} entities")
        
        return final_results if final_results is not None else pd.DataFrame()
    
    def save_model(self, filepath: str):
        """Save the trained anomaly detection models."""
        model_data = {
            'models': self.models,
            'scalers': self.scalers,
            'pattern_baselines': self.pattern_baselines,
            'contamination': self.contamination
        }
        joblib.dump(model_data, filepath)
        logger.info(f"Anomaly detection models saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained anomaly detection models."""
        model_data = joblib.load(filepath)
        self.models = model_data['models']
        self.scalers = model_data['scalers']
        self.pattern_baselines = model_data['pattern_baselines']
        self.contamination = model_data['contamination']
        logger.info(f"Anomaly detection models loaded from {filepath}")

def main():
    """Test anomaly detection models."""
    from data_pipeline import DataPipeline
    from feature_engineering import FeatureEngineer
    
    # Load sample data
    pipeline = DataPipeline("/home/fiod/shimshi/bq-results-20250804-141411-1754316868932.csv")
    df = pipeline.load_data_chunked(sample_fraction=0.05)  # 5% sample
    
    # Engineer features
    feature_engineer = FeatureEngineer()
    features_df = feature_engineer.create_all_features(df)
    
    # Run anomaly detection
    anomaly_detector = AnomalyDetector(contamination=0.1)
    anomaly_results = anomaly_detector.run_comprehensive_detection(features_df)
    
    # Display results
    if not anomaly_results.empty:
        logger.info(f"Anomaly detection results shape: {anomaly_results.shape}")
        
        # Count anomalies by type
        anomaly_cols = [col for col in anomaly_results.columns if 'anomaly' in col]
        for col in anomaly_cols:
            if col in anomaly_results.columns:
                count = anomaly_results[col].sum() if anomaly_results[col].dtype == bool else 0
                logger.info(f"{col}: {count} anomalies detected")
        
        # Show top anomalous channels
        if 'overall_anomaly_count' in anomaly_results.columns:
            top_anomalies = anomaly_results.nlargest(10, 'overall_anomaly_count')
            logger.info("Top 10 most anomalous channels:")
            logger.info(top_anomalies[['overall_anomaly_count', 'overall_anomaly_flag']])
    
    # Save model
    anomaly_detector.save_model("/home/fiod/shimshi/anomaly_detection_model.pkl")
    
    return anomaly_detector, anomaly_results

if __name__ == "__main__":
    anomaly_detector, anomaly_results = main()