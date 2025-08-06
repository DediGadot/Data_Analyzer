"""
Pattern Anomaly Detection Models - Fixed Version
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
        
        # Reset index to make channelId a regular column for easier handling
        country_patterns_reset = country_patterns.reset_index()
        channel_ids = country_patterns_reset['channelId']
        country_data = country_patterns_reset.drop('channelId', axis=1)
        
        # Detect channels with unusual country distributions
        scaler = StandardScaler()
        country_scaled = scaler.fit_transform(country_data)
        
        # Use Isolation Forest to detect unusual geographic patterns
        iso_forest = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state
        )
        geo_anomaly_scores = iso_forest.fit_predict(country_scaled)
        geo_anomaly_scores = iso_forest.decision_function(country_scaled)
        
        geo_results = pd.DataFrame({
            'channelId': channel_ids,
            'geo_anomaly_score': geo_anomaly_scores,
            'geo_is_anomaly': geo_anomaly_scores < 0,
            'country_diversity': country_data.gt(0).sum(axis=1).values,
            'dominant_country_pct': (country_data.max(axis=1) / country_data.sum(axis=1)).values
        })
        
        # IP-based geographic anomalies
        if 'ip' in df.columns:
            ip_geo_anomalies = self._detect_ip_geographic_anomalies(df)
            geo_results = geo_results.merge(ip_geo_anomalies, on='channelId', how='left')
        
        self.models['geographic'] = {
            'iso_forest': iso_forest,
            'scaler': scaler
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
        
        all_features = []
        channel_ids = df['channelId'].unique()
        
        # Device/Browser combination patterns
        if 'device' in df.columns and 'browser' in df.columns:
            # Create pivot table for device-browser combinations
            device_browser_pivot = pd.pivot_table(
                df, 
                values='date', 
                index='channelId', 
                columns=['device', 'browser'], 
                aggfunc='count', 
                fill_value=0
            )
            # Flatten column names
            device_browser_pivot.columns = [f'dev_{d}_br_{b}' for d, b in device_browser_pivot.columns]
            all_features.append(device_browser_pivot)
        
        # User agent pattern analysis
        if 'userAgent' in df.columns:
            ua_features = self._analyze_user_agent_patterns(df)
            if not ua_features.empty:
                ua_features_indexed = ua_features.set_index('channelId')
                all_features.append(ua_features_indexed)
        
        # Browser version anomalies
        if 'browserMajorVersion' in df.columns:
            version_features = self._analyze_browser_version_patterns(df)
            if not version_features.empty:
                version_features_indexed = version_features.set_index('channelId')
                all_features.append(version_features_indexed)
        
        if not all_features:
            logger.warning("No device data available for anomaly detection")
            return pd.DataFrame()
        
        # Combine all device features
        if len(all_features) == 1:
            combined_features = all_features[0]
        else:
            # Join all features on index (channelId)
            combined_features = all_features[0]
            for feat in all_features[1:]:
                combined_features = combined_features.join(feat, how='outer')
        
        combined_features = combined_features.fillna(0)
        
        # Ensure we have all channels
        combined_features = combined_features.reindex(channel_ids, fill_value=0)
        
        # Remove constant features
        non_constant = combined_features.std() > 0
        combined_features = combined_features.loc[:, non_constant]
        
        if combined_features.empty or len(combined_features) < 2:
            return pd.DataFrame({'channelId': channel_ids})
        
        # Detect anomalies
        scaler = RobustScaler()
        features_scaled = scaler.fit_transform(combined_features)
        
        # Multiple anomaly detection methods
        detectors = {
            'isolation_forest': IsolationForest(contamination=self.contamination, random_state=self.random_state),
            'lof': LocalOutlierFactor(contamination=self.contamination),
            'one_class_svm': OneClassSVM(nu=self.contamination)
        }
        
        anomaly_scores = {}
        for name, detector in detectors.items():
            try:
                if name == 'lof':
                    # LOF returns -1 for outliers, 1 for inliers
                    anomaly_scores[name] = detector.fit_predict(features_scaled)
                else:
                    anomaly_scores[name] = detector.fit_predict(features_scaled)
            except Exception as e:
                logger.warning(f"Device anomaly detection with {name} failed: {e}")
                anomaly_scores[name] = np.zeros(len(combined_features))
        
        # Ensemble approach: flag as anomaly if multiple methods agree
        anomaly_ensemble = sum(score == -1 for score in anomaly_scores.values()) >= 2
        
        device_results = pd.DataFrame({
            'channelId': combined_features.index,
            'device_isolation_forest_anomaly': anomaly_scores.get('isolation_forest', np.zeros(len(combined_features))) == -1,
            'device_lof_anomaly': anomaly_scores.get('lof', np.zeros(len(combined_features))) == -1,
            'device_one_class_svm_anomaly': anomaly_scores.get('one_class_svm', np.zeros(len(combined_features))) == -1,
            'device_anomaly_ensemble': anomaly_ensemble
        })
        
        self.models['device'] = {
            'detectors': detectors,
            'scaler': scaler
        }
        
        return device_results
    
    def detect_behavioral_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect behavioral pattern anomalies using multiple approaches.
        
        Args:
            df: DataFrame with traffic data
            
        Returns:
            DataFrame with behavioral anomaly scores
        """
        logger.info("Detecting behavioral anomalies")
        
        # Extract behavioral features
        behavioral_features = self._extract_behavioral_features(df)
        
        if behavioral_features.empty:
            return pd.DataFrame()
        
        # Scale features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(behavioral_features)
        
        # Multiple anomaly detection approaches
        iso_forest = IsolationForest(contamination=self.contamination, random_state=self.random_state)
        iso_anomalies = iso_forest.fit_predict(features_scaled) == -1
        
        elliptic = EllipticEnvelope(contamination=self.contamination, random_state=self.random_state)
        elliptic_anomalies = elliptic.fit_predict(features_scaled) == -1
        
        svm = OneClassSVM(nu=self.contamination)
        svm_anomalies = svm.fit_predict(features_scaled) == -1
        
        # Ensemble approach
        ensemble_score = iso_anomalies.astype(int) + elliptic_anomalies.astype(int) + svm_anomalies.astype(int)
        
        behavioral_results = pd.DataFrame({
            'channelId': behavioral_features.index,
            'behavioral_isolation_forest_anomaly': iso_anomalies,
            'behavioral_elliptic_envelope_anomaly': elliptic_anomalies,
            'behavioral_one_class_svm_anomaly': svm_anomalies,
            'behavioral_anomaly_ensemble': ensemble_score >= 2
        })
        
        self.models['behavioral'] = {
            'iso_forest': iso_forest,
            'elliptic': elliptic,
            'svm': svm,
            'scaler': scaler
        }
        
        return behavioral_results
    
    def detect_volume_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect volume-based anomalies at various levels.
        
        Args:
            df: DataFrame with traffic data
            
        Returns:
            DataFrame with volume anomaly scores
        """
        logger.info("Detecting volume anomalies")
        
        volume_features = []
        
        # Channel-level volume patterns
        if 'channelId' in df.columns:
            channel_volumes = df.groupby('channelId').agg({
                'date': 'count',
                'ip': 'nunique',
                'userId': lambda x: x.nunique() if 'userId' in df.columns else 0
            })
            channel_volumes.columns = ['total_requests', 'unique_ips', 'unique_users']
            volume_features.append(channel_volumes)
        
        # IP-level volume patterns (detecting IP flooding)
        if 'ip' in df.columns:
            ip_volumes = df.groupby('ip').agg({
                'date': 'count',
                'channelId': 'nunique'
            })
            ip_volumes.columns = ['requests_per_ip', 'channels_per_ip']
            
            # Flag IPs with unusually high request volumes
            ip_threshold = ip_volumes['requests_per_ip'].quantile(0.95)
            suspicious_ips = ip_volumes[ip_volumes['requests_per_ip'] > ip_threshold].index
            
            # Aggregate back to channel level
            channel_suspicious_ips = df[df['ip'].isin(suspicious_ips)].groupby('channelId').size()
            channel_suspicious_ips = channel_suspicious_ips.reindex(channel_volumes.index, fill_value=0)
            
            volume_features.append(pd.DataFrame({'suspicious_ip_requests': channel_suspicious_ips}))
        
        if not volume_features:
            return pd.DataFrame()
        
        # Combine volume features
        combined_volumes = pd.concat(volume_features, axis=1).fillna(0)
        
        # Detect anomalies using statistical methods
        volume_anomalies = {}
        
        for col in combined_volumes.columns:
            # Z-score based anomaly detection
            z_scores = np.abs(stats.zscore(combined_volumes[col]))
            volume_anomalies[f'{col}_anomaly'] = z_scores > 3
        
        volume_results = pd.DataFrame(volume_anomalies, index=combined_volumes.index)
        volume_results['channelId'] = combined_volumes.index
        
        return volume_results.reset_index(drop=True)
    
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
            'channelId': temporal_features.index,
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
        z_scores = np.abs(stats.zscore(daily_patterns, axis=1))
        daily_anomalies = (z_scores > 2).any(axis=1)
        
        results = pd.DataFrame({
            'channelId': temporal_features.index,
            'daily_anomaly': daily_anomalies,
            'daily_max_z_score': z_scores.max(axis=1)
        })
        
        return results
    
    def _detect_traffic_bursts(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect sudden bursts in traffic patterns."""
        df['datetime'] = pd.to_datetime(df['date'])
        
        burst_detection = []
        
        for entity_col in ['channelId', 'ip']:
            if entity_col not in df.columns:
                continue
            
            for entity in df[entity_col].unique():
                entity_data = df[df[entity_col] == entity].copy()
                entity_data = entity_data.sort_values('datetime')
                
                # Calculate time differences between requests
                time_diffs = entity_data['datetime'].diff().dt.total_seconds()
                
                # Detect bursts: many requests in short time
                if len(time_diffs) > 5:
                    # Calculate rolling statistics
                    rolling_mean = time_diffs.rolling(window=5, center=True).mean()
                    rolling_std = time_diffs.rolling(window=5, center=True).std()
                    
                    # Flag bursts: time between requests significantly lower than normal
                    burst_mask = time_diffs < (rolling_mean - 2 * rolling_std)
                    
                    if burst_mask.any():
                        burst_detection.append({
                            entity_col: entity,
                            'burst_count': burst_mask.sum(),
                            'min_time_diff': time_diffs[burst_mask].min() if burst_mask.any() else np.nan
                        })
        
        if burst_detection:
            burst_df = pd.DataFrame(burst_detection)
            # Aggregate to channel level if needed
            if 'channelId' in burst_df.columns:
                return burst_df[burst_df['channelId'].notna()].groupby('channelId').agg({
                    'burst_count': 'sum',
                    'min_time_diff': 'min'
                }).reset_index()
        
        return pd.DataFrame()
    
    def _detect_ip_geographic_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect IP-based geographic anomalies."""
        # IPs appearing in multiple countries (potential proxy/VPN indicators)
        ip_countries = df.groupby('ip')['country'].nunique()
        multi_country_ips = ip_countries[ip_countries > 1]
        
        # Channel-level IP geographic diversity anomalies
        channel_ip_geo = df.groupby('channelId').apply(
            lambda x: pd.Series({
                'multi_country_ips': x['ip'].isin(multi_country_ips.index).sum(),
                'country_diversity': x['country'].nunique()
            })
        )
        
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
            lambda x: x.str.len().std(),
            lambda x: (x.str.len() < 50).mean(),  # Suspiciously short UAs
            lambda x: x.value_counts().iloc[0] / len(x) if len(x) > 0 else 0  # Dominant UA percentage
        ])
        ua_analysis.columns = ['ua_diversity', 'ua_avg_length', 'ua_length_std', 'short_ua_rate', 'dominant_ua_pct']
        
        return ua_analysis.reset_index()
    
    def _analyze_browser_version_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Analyze browser version patterns."""
        if 'browserMajorVersion' not in df.columns:
            return pd.DataFrame()
        
        version_analysis = df.groupby('channelId')['browserMajorVersion'].agg([
            'nunique',
            'mean',
            'std',
            lambda x: (x < 50).mean(),  # Outdated browser rate
            lambda x: x.mode().iloc[0] if len(x) > 0 and len(x.mode()) > 0 else 0
        ])
        version_analysis.columns = ['version_diversity', 'avg_version', 'version_std', 'outdated_browser_rate', 'most_common_version']
        
        return version_analysis.reset_index()
    
    def _extract_behavioral_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract behavioral features for anomaly detection."""
        behavioral_features = []
        
        # Request timing patterns
        if 'date' in df.columns:
            df_sorted = df.sort_values(['channelId', 'date'])
            timing_features = df_sorted.groupby('channelId').agg({
                'date': [
                    lambda x: x.diff().dt.total_seconds().mean(),  # Average time between requests
                    lambda x: x.diff().dt.total_seconds().std(),   # Variance in timing
                    lambda x: (x.diff().dt.total_seconds() < 1).sum(),  # Rapid fire requests
                ]
            })
            timing_features.columns = ['avg_time_between_requests', 'time_variance', 'rapid_requests']
            behavioral_features.append(timing_features)
        
        # Keyword diversity and patterns
        if 'keyword' in df.columns:
            keyword_features = df.groupby('channelId')['keyword'].agg([
                'nunique',
                lambda x: x.str.len().mean(),
                lambda x: x.value_counts().iloc[0] / len(x) if len(x) > 0 else 0
            ])
            keyword_features.columns = ['keyword_diversity', 'avg_keyword_length', 'dominant_keyword_pct']
            behavioral_features.append(keyword_features)
        
        # Referrer patterns
        if 'referrer' in df.columns:
            referrer_features = df.groupby('channelId')['referrer'].agg([
                'nunique',
                lambda x: x.isna().mean()
            ])
            referrer_features.columns = ['referrer_diversity', 'no_referrer_rate']
            behavioral_features.append(referrer_features)
        
        if not behavioral_features:
            return pd.DataFrame()
        
        # Combine all behavioral features
        combined_features = pd.concat(behavioral_features, axis=1).fillna(0)
        
        return combined_features
    
    def _combine_anomaly_scores(self, anomaly_results: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Combine different anomaly detection results."""
        if not anomaly_results:
            return pd.DataFrame()
        
        # Merge all results on channelId
        combined_results = None
        for anomaly_type, results in anomaly_results.items():
            if not results.empty:
                if combined_results is None:
                    combined_results = results.copy()
                else:
                    combined_results = combined_results.merge(results, on='channelId', how='outer')
        
        return combined_results.fillna(False) if combined_results is not None else pd.DataFrame()
    
    def run_comprehensive_anomaly_detection(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run all anomaly detection methods and combine results.
        
        Args:
            df: DataFrame with traffic data
            
        Returns:
            DataFrame with all anomaly scores and flags
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
            anomaly_cols = [col for col in final_results.columns if 'anomaly' in col and 'overall' not in col and '_score' not in col]
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
    
    @classmethod
    def load_model(cls, filepath: str) -> 'AnomalyDetector':
        """Load a saved anomaly detection model."""
        model_data = joblib.load(filepath)
        
        detector = cls(contamination=model_data['contamination'])
        detector.models = model_data['models']
        detector.scalers = model_data['scalers']
        detector.pattern_baselines = model_data['pattern_baselines']
        
        logger.info(f"Anomaly detection models loaded from {filepath}")
        return detector