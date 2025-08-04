"""
Feature Engineering Pipeline for Fraud Detection
Advanced feature extraction from advertising traffic data for ML models.
"""

import pandas as pd
import numpy as np
import re
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import hashlib
from urllib.parse import urlparse
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import ipaddress
import logging

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """
    Advanced feature engineering for fraud detection in advertising traffic.
    Creates temporal, behavioral, IP-based, and contextual features.
    """
    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.vectorizers = {}
        
    def create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create all feature sets for fraud detection models.
        
        Args:
            df: Input DataFrame with raw traffic data
            
        Returns:
            DataFrame with engineered features
        """
        logger.info("Starting comprehensive feature engineering")
        
        # Make a copy to avoid modifying original data
        features_df = df.copy()
        
        # Temporal features
        features_df = self._create_temporal_features(features_df)
        
        # IP-based features
        features_df = self._create_ip_features(features_df)
        
        # User agent features
        features_df = self._create_user_agent_features(features_df)
        
        # Behavioral features
        features_df = self._create_behavioral_features(features_df)
        
        # Contextual features
        features_df = self._create_contextual_features(features_df)
        
        # Fraud indicator aggregations
        features_df = self._create_fraud_aggregations(features_df)
        
        # Traffic pattern features
        features_df = self._create_traffic_patterns(features_df)
        
        logger.info(f"Feature engineering complete. Created {features_df.shape[1] - df.shape[1]} new features")
        
        return features_df
    
    def _create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features."""
        logger.info("Creating temporal features")
        
        # Ensure date is datetime - handle mixed formats
        df['date'] = pd.to_datetime(df['date'], format='mixed')
        
        # Basic time components
        df['hour'] = df['date'].dt.hour
        df['day_of_week'] = df['date'].dt.dayofweek
        df['day_of_month'] = df['date'].dt.day
        df['month'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter
        
        # Time-based patterns
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 6)).astype(int)
        df['is_business_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 17) & ~df['is_weekend']).astype(int)
        
        # Time since epoch (for trend analysis)
        df['timestamp'] = df['date'].astype(np.int64) // 10**9
        
        # Seasonal patterns
        df['sin_hour'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['cos_hour'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['sin_day_of_week'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['cos_day_of_week'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        return df
    
    def _create_ip_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create IP-based features."""
        logger.info("Creating IP-based features")
        
        # IP type classification
        df['ip_type'] = df['ip'].apply(self._classify_ip_type)
        
        # IP network features
        df['ip_network_class'] = df['ip'].apply(self._get_ip_network_class)
        
        # Private/Public IP classification
        df['is_private_ip'] = df['ip'].apply(self._is_private_ip)
        
        # IP geographic consistency (requires external service in production)
        df['ip_country_match'] = 1  # Placeholder - would need IP geolocation service
        
        # IP entropy (measure of randomness)
        df['ip_entropy'] = df['ip'].apply(self._calculate_ip_entropy)
        
        # Aggregate IP-based fraud indicators
        fraud_ip_cols = [col for col in df.columns if col.startswith('isIp') and col != 'ip']
        if fraud_ip_cols:
            df['total_ip_flags'] = df[fraud_ip_cols].sum(axis=1)
            df['ip_risk_score'] = df['total_ip_flags'] / len(fraud_ip_cols)
        
        return df
    
    def _create_user_agent_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract features from user agent strings."""
        logger.info("Creating user agent features")
        
        if 'userAgent' not in df.columns:
            return df
            
        # User agent length and complexity
        df['ua_length'] = df['userAgent'].astype(str).str.len()
        df['ua_word_count'] = df['userAgent'].astype(str).str.split().str.len()
        
        # Browser version consistency
        df['browser_version_match'] = self._check_browser_version_consistency(df)
        
        # Common bot patterns in user agents
        bot_patterns = [
            'bot', 'crawler', 'spider', 'scraper', 'curl', 'wget', 
            'python', 'java', 'scrapy', 'requests'
        ]
        df['ua_bot_indicators'] = df['userAgent'].astype(str).str.lower().str.contains(
            '|'.join(bot_patterns), na=False
        ).astype(int)
        
        # User agent entropy
        df['ua_entropy'] = df['userAgent'].apply(self._calculate_string_entropy)
        
        # Suspicious patterns
        df['ua_has_version'] = df['userAgent'].astype(str).str.contains(r'\d+\.\d+', na=False).astype(int)
        df['ua_suspicious'] = (
            (df['ua_length'] < 10) | 
            (df['ua_length'] > 500) |
            (df['ua_word_count'] < 2)
        ).astype(int)
        
        return df
    
    def _create_behavioral_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create behavioral pattern features."""
        logger.info("Creating behavioral features")
        
        # Aggregate by different time windows
        for entity in ['channelId', 'publisherId', 'ip', 'userId']:
            if entity in df.columns:
                # Request frequency
                df[f'{entity}_daily_requests'] = df.groupby([entity, df['date'].dt.date])['date'].transform('count')
                df[f'{entity}_hourly_requests'] = df.groupby([entity, df['date'].dt.floor('H')])['date'].transform('count')
                
                # Country diversity
                df[f'{entity}_country_diversity'] = df.groupby(entity)['country'].transform('nunique')
                
                # Time pattern consistency
                df[f'{entity}_hour_std'] = df.groupby(entity)['hour'].transform('std').fillna(0)
                
        # Channel-specific patterns
        if 'channelId' in df.columns:
            channel_stats = df.groupby('channelId').agg({
                'isLikelyBot': ['mean', 'count'],
                'country': 'nunique',
                'browser': 'nunique',
                'device': 'nunique',
                'hour': 'std'
            }).round(4)
            
            channel_stats.columns = [f'channel_{col[0]}_{col[1]}' for col in channel_stats.columns]
            df = df.merge(channel_stats, left_on='channelId', right_index=True, how='left')
        
        return df
    
    def _create_contextual_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create contextual features from referrers, keywords, etc."""
        logger.info("Creating contextual features")
        
        # Referrer analysis
        if 'referrer' in df.columns:
            df['referrer_domain'] = df['referrer'].apply(self._extract_domain)
            df['referrer_is_social'] = df['referrer_domain'].isin([
                'facebook.com', 'twitter.com', 'instagram.com', 'linkedin.com',
                'tiktok.com', 'youtube.com', 'pinterest.com'
            ]).astype(int)
            df['referrer_is_search'] = df['referrer_domain'].isin([
                'google.com', 'bing.com', 'yahoo.com', 'duckduckgo.com'
            ]).astype(int)
            df['referrer_is_ad_network'] = df['referrer_domain'].isin([
                'taboola.com', 'outbrain.com', 'googleadservices.com'
            ]).astype(int)
        
        # Keyword analysis
        if 'keyword' in df.columns:
            df['keyword_length'] = df['keyword'].astype(str).str.len()
            df['keyword_word_count'] = df['keyword'].astype(str).str.split().str.len()
            df['keyword_has_brand'] = df['keyword'].astype(str).str.contains(
                r'samsung|apple|microsoft|google|amazon', case=False, na=False
            ).astype(int)
            
            # Keyword commercial intent indicators
            commercial_terms = ['buy', 'price', 'deal', 'discount', 'sale', 'cheap', 'best']
            df['keyword_commercial_intent'] = df['keyword'].astype(str).str.lower().str.contains(
                '|'.join(commercial_terms), na=False
            ).astype(int)
        
        # Device and browser combinations
        if 'device' in df.columns and 'browser' in df.columns:
            df['device_browser_combo'] = df['device'].astype(str) + '_' + df['browser'].astype(str)
            combo_counts = df['device_browser_combo'].value_counts()
            df['device_browser_popularity'] = df['device_browser_combo'].map(combo_counts)
        
        return df
    
    def _create_fraud_aggregations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create aggregated fraud indicators."""
        logger.info("Creating fraud aggregations")
        
        # List all boolean fraud indicators
        fraud_cols = [col for col in df.columns if col.startswith('isIp') or col in ['isLikelyBot']]
        
        if fraud_cols:
            # Total fraud flags per record
            df['total_fraud_flags'] = df[fraud_cols].sum(axis=1)
            
            # Weighted fraud score (some indicators are more serious)
            fraud_weights = {
                'isLikelyBot': 3.0,
                'isIpTOR': 2.5,
                'isIpDatacenter': 2.0,
                'isIpVPN': 1.5,
                'isIpAnonymous': 1.5,
                'isIpPublicProxy': 1.8,
                'isIpHostingService': 2.0,
                'isIpResidentialProxy': 1.2,
                'isIpCrawler': 2.2
            }
            
            df['weighted_fraud_score'] = sum(
                df.get(col, 0) * weight 
                for col, weight in fraud_weights.items()
                if col in df.columns
            )
        
        return df
    
    def _create_traffic_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create traffic pattern features for anomaly detection."""
        logger.info("Creating traffic pattern features")
        
        # Sort by timestamp for sequence analysis
        df = df.sort_values('date')
        
        # Time gaps between requests (by IP/User)
        for entity in ['ip', 'userId']:
            if entity in df.columns:
                df[f'{entity}_time_gap'] = df.groupby(entity)['timestamp'].diff().fillna(0)
                df[f'{entity}_request_velocity'] = 1 / (df[f'{entity}_time_gap'] + 1)  # Requests per second
        
        # Burst detection (many requests in short time)
        if 'ip' in df.columns:
            df['ip_burst_5min'] = df.groupby('ip')['date'].transform(
                lambda x: x.diff().dt.total_seconds().rolling(5, min_periods=1).sum() < 300
            ).astype(int)
        
        # Geographic anomalies
        if 'country' in df.columns and 'ip' in df.columns:
            ip_countries = df.groupby('ip')['country'].nunique()
            df['ip_country_switches'] = df['ip'].map(ip_countries) - 1
        
        return df
    
    # Helper methods
    def _classify_ip_type(self, ip_str: str) -> str:
        """Classify IP address type."""
        try:
            ip = ipaddress.ip_address(ip_str)
            if ip.version == 4:
                return 'ipv4'
            elif ip.version == 6:
                return 'ipv6'
        except:
            return 'invalid'
        return 'unknown'
    
    def _get_ip_network_class(self, ip_str: str) -> str:
        """Get IP network class (A, B, C for IPv4)."""
        try:
            ip = ipaddress.IPv4Address(ip_str)
            first_octet = int(str(ip).split('.')[0])
            if first_octet <= 127:
                return 'A'
            elif first_octet <= 191:
                return 'B'
            elif first_octet <= 223:
                return 'C'
            else:
                return 'Other'
        except:
            return 'Unknown'
    
    def _is_private_ip(self, ip_str: str) -> int:
        """Check if IP is private."""
        try:
            ip = ipaddress.ip_address(ip_str)
            return int(ip.is_private)
        except:
            return 0
    
    def _calculate_ip_entropy(self, ip_str: str) -> float:
        """Calculate entropy of IP address."""
        return self._calculate_string_entropy(ip_str)
    
    def _calculate_string_entropy(self, s: str) -> float:
        """Calculate Shannon entropy of a string."""
        if not s or pd.isna(s):
            return 0.0
        
        s = str(s)
        prob = [float(s.count(c)) / len(s) for c in dict.fromkeys(list(s))]
        entropy = -sum([p * np.log2(p) for p in prob])
        return entropy
    
    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL."""
        try:
            if pd.isna(url) or not url:
                return 'unknown'
            parsed = urlparse(str(url))
            domain = parsed.netloc.lower()
            # Remove www. prefix
            if domain.startswith('www.'):
                domain = domain[4:]
            return domain
        except:
            return 'unknown'
    
    def _check_browser_version_consistency(self, df: pd.DataFrame) -> pd.Series:
        """Check consistency between browser and browserMajorVersion."""
        if 'browser' not in df.columns or 'browserMajorVersion' not in df.columns:
            return pd.Series([1] * len(df))
        
        # This is a simplified check - in production, you'd have a comprehensive mapping
        consistency = pd.Series([1] * len(df))
        
        # Check for obvious inconsistencies
        chrome_mask = df['browser'] == 'chrome'
        if chrome_mask.any():
            # Chrome versions should be reasonable (e.g., 70-120)
            consistency.loc[chrome_mask & ((df['browserMajorVersion'] < 70) | 
                                         (df['browserMajorVersion'] > 130))] = 0
        
        return consistency
    
    def create_channel_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create channel-level aggregated features for similarity analysis."""
        logger.info("Creating channel-level features")
        
        if 'channelId' not in df.columns:
            return pd.DataFrame()
        
        # Aggregate features by channel
        channel_features = df.groupby('channelId').agg({
            # Volume metrics
            'date': 'count',
            
            # Fraud indicators
            'isLikelyBot': ['mean', 'sum'],
            'total_fraud_flags': ['mean', 'sum'],
            'weighted_fraud_score': ['mean', 'max'],
            
            # Diversity metrics
            'country': 'nunique',
            'browser': 'nunique',
            'device': 'nunique',
            'ip': 'nunique',
            
            # Temporal patterns
            'hour': ['mean', 'std'],
            'day_of_week': ['mean', 'std'],
            'is_weekend': 'mean',
            'is_night': 'mean',
            'is_business_hours': 'mean',
            
            # Technical patterns
            'ua_length': 'mean',
            'ua_entropy': 'mean',
            'ip_entropy': 'mean',
            
            # Behavioral patterns
            'keyword_length': 'mean',
            'keyword_commercial_intent': 'mean'
        }).round(4)
        
        # Flatten column names
        channel_features.columns = [f'channel_{col[0]}_{col[1]}' for col in channel_features.columns]
        
        # Add ratio metrics
        channel_features['channel_bot_ratio'] = (
            channel_features['channel_isLikelyBot_sum'] / 
            channel_features['channel_date_count']
        ).round(4)
        
        channel_features['channel_unique_ip_ratio'] = (
            channel_features['channel_ip_nunique'] / 
            channel_features['channel_date_count']
        ).round(4)
        
        return channel_features

def main():
    """Test feature engineering pipeline."""
    from data_pipeline import DataPipeline
    
    # Load sample data
    pipeline = DataPipeline("/home/fiod/shimshi/bq-results-20250804-141411-1754316868932.csv")
    df = pipeline.load_data_chunked(sample_fraction=0.01)  # 1% sample for testing
    
    # Create features
    feature_engineer = FeatureEngineer()
    features_df = feature_engineer.create_all_features(df)
    
    # Create channel features
    channel_features = feature_engineer.create_channel_features(features_df)
    
    logger.info(f"Original features: {df.shape[1]}")
    logger.info(f"Engineered features: {features_df.shape[1]}")
    logger.info(f"Channel features: {channel_features.shape}")
    
    return features_df, channel_features

if __name__ == "__main__":
    features_df, channel_features = main()