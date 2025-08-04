"""
High-Performance Traffic Dataset Analyzer
==========================================

A comprehensive system for processing and analyzing large traffic datasets (1.4M+ records)
with focus on fraud detection, performance optimization, and advanced analytics.

Features:
- Memory-efficient chunked data loading
- Advanced feature extraction from JSON fields
- Temporal and behavioral pattern analysis
- Channel similarity calculations
- Quality scoring and anomaly detection
"""

import json
import logging
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import lru_cache, wraps
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple, Union
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import polars as pl
from scipy.spatial.distance import cosine
from scipy.stats import zscore
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from user_agents import parse as parse_user_agent

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def timing_decorator(func):
    """Decorator to measure function execution time."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        logger.info(f"{func.__name__} executed in {end_time - start_time:.4f} seconds")
        return result
    return wrapper


@dataclass
class DataLoadConfig:
    """Configuration for data loading optimization."""
    chunk_size: int = 50000
    use_polars: bool = True
    memory_map: bool = True
    low_memory: bool = True
    engine: str = 'c'
    dtype_optimization: bool = True


@dataclass
class FeatureConfig:
    """Configuration for feature extraction."""
    extract_platform: bool = True
    extract_location: bool = True
    parse_user_agent: bool = True
    temporal_features: bool = True
    ip_features: bool = True


class TrafficDataLoader:
    """Memory-efficient data loader with chunked processing and optimal data types."""
    
    def __init__(self, config: DataLoadConfig = None):
        self.config = config or DataLoadConfig()
        self._optimal_dtypes = self._get_optimal_dtypes()
    
    def _get_optimal_dtypes(self) -> Dict[str, str]:
        """Define optimal data types for memory efficiency."""
        return {
            'publisherId': 'string',
            'channelId': 'string', 
            'advertiserId': 'string',
            'feedId': 'string',
            'keyword': 'string',
            'country': 'category',
            'browser': 'category',
            'device': 'category',
            'referrer': 'string',
            'ip': 'string',
            'browserMajorVersion': 'int16',
            'userId': 'string',
            'isLikelyBot': 'bool',
            'ipClassification': 'category',
            'isIpDatacenter': 'bool',
            'datacenterName': 'string',
            'ipHostName': 'string',
            'isIpAnonymous': 'bool',
            'isIpCrawler': 'bool',
            'isIpPublicProxy': 'bool',
            'isIpVPN': 'bool',
            'isIpHostingService': 'bool',
            'isIpTOR': 'bool',
            'isIpResidentialProxy': 'bool',
            'performance': 'string',
            'detection': 'string',
            'platform': 'string',
            'location': 'string',
            'userAgent': 'string'
        }
    
    @timing_decorator
    def load_data_pandas(self, file_path: str, nrows: Optional[int] = None) -> pd.DataFrame:
        """Load data using pandas with memory optimizations."""
        logger.info(f"Loading data from {file_path} using pandas...")
        
        # Parse dates efficiently
        date_parser = lambda x: pd.to_datetime(x, format='%Y-%m-%d %H:%M:%S.%f UTC', errors='coerce')
        
        df = pd.read_csv(
            file_path,
            dtype=self._optimal_dtypes,
            parse_dates=['date'],
            date_parser=date_parser,
            nrows=nrows,
            engine=self.config.engine,
            low_memory=self.config.low_memory,
            memory_map=self.config.memory_map
        )
        
        if self.config.dtype_optimization:
            df = self._optimize_memory_usage(df)
        
        logger.info(f"Loaded {len(df):,} records. Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        return df
    
    @timing_decorator
    def load_data_polars(self, file_path: str, nrows: Optional[int] = None) -> pl.DataFrame:
        """Load data using polars for maximum performance."""
        logger.info(f"Loading data from {file_path} using polars...")
        
        # Define polars schema for optimal types
        schema = {
            'date': pl.Datetime,
            'publisherId': pl.Utf8,
            'channelId': pl.Utf8,
            'advertiserId': pl.Utf8,
            'feedId': pl.Utf8,
            'keyword': pl.Utf8,
            'country': pl.Categorical,
            'browser': pl.Categorical,
            'device': pl.Categorical,
            'referrer': pl.Utf8,
            'ip': pl.Utf8,
            'browserMajorVersion': pl.Int16,
            'userId': pl.Utf8,
            'isLikelyBot': pl.Boolean,
            'ipClassification': pl.Categorical,
            'isIpDatacenter': pl.Boolean,
            'datacenterName': pl.Utf8,
            'ipHostName': pl.Utf8,
            'isIpAnonymous': pl.Boolean,
            'isIpCrawler': pl.Boolean,
            'isIpPublicProxy': pl.Boolean,
            'isIpVPN': pl.Boolean,
            'isIpHostingService': pl.Boolean,
            'isIpTOR': pl.Boolean,
            'isIpResidentialProxy': pl.Boolean,
            'performance': pl.Utf8,
            'detection': pl.Utf8,
            'platform': pl.Utf8,
            'location': pl.Utf8,
            'userAgent': pl.Utf8
        }
        
        df = pl.read_csv(
            file_path,
            schema=schema,
            n_rows=nrows,
            try_parse_dates=True
        )
        
        logger.info(f"Loaded {len(df):,} records using polars")
        return df
    
    def load_data_chunked(self, file_path: str, chunk_processor=None) -> Generator[pd.DataFrame, None, None]:
        """Generator for chunked data processing to handle large files."""
        logger.info(f"Loading data in chunks of {self.config.chunk_size:,} records...")
        
        chunk_count = 0
        for chunk in pd.read_csv(
            file_path,
            dtype=self._optimal_dtypes,
            parse_dates=['date'],
            chunksize=self.config.chunk_size,
            engine=self.config.engine,
            low_memory=self.config.low_memory
        ):
            chunk_count += 1
            if self.config.dtype_optimization:
                chunk = self._optimize_memory_usage(chunk)
            
            if chunk_processor:
                chunk = chunk_processor(chunk)
            
            logger.info(f"Processing chunk {chunk_count} with {len(chunk):,} records")
            yield chunk
    
    def _optimize_memory_usage(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize memory usage by downcasting numeric types."""
        start_memory = df.memory_usage(deep=True).sum() / 1024**2
        
        for col in df.columns:
            col_type = df[col].dtype
            
            if col_type != object:
                c_min = df[col].min()
                c_max = df[col].max()
                
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)
                        
                elif str(col_type)[:5] == 'float':
                    if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                    else:
                        df[col] = df[col].astype(np.float64)
        
        end_memory = df.memory_usage(deep=True).sum() / 1024**2
        logger.info(f'Memory usage decreased from {start_memory:.2f} MB to {end_memory:.2f} MB '
                   f'({100 * (start_memory - end_memory) / start_memory:.1f}% reduction)')
        
        return df
    
    def load_data(self, file_path: str, nrows: Optional[int] = None) -> Union[pd.DataFrame, pl.DataFrame]:
        """Main data loading method that chooses optimal backend."""
        if self.config.use_polars:
            return self.load_data_polars(file_path, nrows)
        else:
            return self.load_data_pandas(file_path, nrows)


class AdvancedFeatureExtractor:
    """Advanced feature extraction from JSON fields, user agents, and other complex data."""
    
    def __init__(self, config: FeatureConfig = None):
        self.config = config or FeatureConfig()
        self._browser_versions_cache = {}
        self._location_cache = {}
    
    @timing_decorator
    def extract_all_features(self, df: Union[pd.DataFrame, pl.DataFrame]) -> pd.DataFrame:
        """Extract all configured features from the dataset."""
        if isinstance(df, pl.DataFrame):
            df = df.to_pandas()
        
        logger.info("Extracting advanced features...")
        
        # Create a copy to avoid modifying original data
        enriched_df = df.copy()
        
        if self.config.extract_platform:
            enriched_df = self._extract_platform_features(enriched_df)
        
        if self.config.extract_location:
            enriched_df = self._extract_location_features(enriched_df)
        
        if self.config.parse_user_agent:
            enriched_df = self._extract_user_agent_features(enriched_df)
        
        if self.config.temporal_features:
            enriched_df = self._extract_temporal_features(enriched_df)
        
        if self.config.ip_features:
            enriched_df = self._extract_ip_features(enriched_df)
        
        logger.info(f"Feature extraction complete. Added {len(enriched_df.columns) - len(df.columns)} new features")
        return enriched_df
    
    def _extract_platform_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract structured features from platform JSON field."""
        logger.info("Extracting platform features...")
        
        def parse_platform(platform_json):
            if pd.isna(platform_json) or platform_json == '':
                return {
                    'cpu_architecture': None,
                    'engine_name': None,
                    'engine_version': None,
                    'os_name': None,
                    'os_version': None
                }
            
            try:
                platform_data = json.loads(platform_json)
                return {
                    'cpu_architecture': platform_data.get('cpu', {}).get('architecture'),
                    'engine_name': platform_data.get('engine', {}).get('name'),
                    'engine_version': platform_data.get('engine', {}).get('version'),
                    'os_name': platform_data.get('os', {}).get('name'),
                    'os_version': platform_data.get('os', {}).get('version')
                }
            except (json.JSONDecodeError, AttributeError):
                return {
                    'cpu_architecture': None,
                    'engine_name': None,
                    'engine_version': None,
                    'os_name': None,
                    'os_version': None
                }
        
        # Extract platform features
        platform_features = df['platform'].apply(parse_platform)
        platform_df = pd.DataFrame(platform_features.tolist())
        
        # Convert to categorical for memory efficiency
        for col in platform_df.columns:
            platform_df[col] = platform_df[col].astype('category')
        
        return pd.concat([df, platform_df], axis=1)
    
    def _extract_location_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract structured features from location JSON field."""
        logger.info("Extracting location features...")
        
        def parse_location(location_json):
            if pd.isna(location_json) or location_json == '':
                return {
                    'city_name': None,
                    'country_code': None,
                    'timezone': None,
                    'timezone_offset': None
                }
            
            try:
                location_data = json.loads(location_json)
                return {
                    'city_name': location_data.get('cityName'),
                    'country_code': location_data.get('countryCode'),
                    'timezone': location_data.get('timezone'),
                    'timezone_offset': location_data.get('timezoneOffset')
                }
            except (json.JSONDecodeError, AttributeError):
                return {
                    'city_name': None,
                    'country_code': None,
                    'timezone': None,
                    'timezone_offset': None
                }
        
        # Extract location features
        location_features = df['location'].apply(parse_location)
        location_df = pd.DataFrame(location_features.tolist())
        
        # Optimize data types
        location_df['city_name'] = location_df['city_name'].astype('category')
        location_df['country_code'] = location_df['country_code'].astype('category')
        location_df['timezone'] = location_df['timezone'].astype('category')
        location_df['timezone_offset'] = pd.to_numeric(location_df['timezone_offset'], errors='coerce').astype('Int16')
        
        return pd.concat([df, location_df], axis=1)
    
    def _extract_user_agent_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract detailed features from user agent strings."""
        logger.info("Extracting user agent features...")
        
        def parse_ua(user_agent):
            if pd.isna(user_agent) or user_agent == '':
                return {
                    'ua_browser_family': None,
                    'ua_browser_version': None,
                    'ua_os_family': None,
                    'ua_os_version': None,
                    'ua_device_family': None,
                    'ua_device_brand': None,
                    'ua_is_mobile': False,
                    'ua_is_tablet': False,
                    'ua_is_pc': False,
                    'ua_is_bot': False
                }
            
            try:
                parsed = parse_user_agent(user_agent)
                return {
                    'ua_browser_family': parsed.browser.family,
                    'ua_browser_version': parsed.browser.version_string,
                    'ua_os_family': parsed.os.family,
                    'ua_os_version': parsed.os.version_string,
                    'ua_device_family': parsed.device.family,
                    'ua_device_brand': parsed.device.brand,
                    'ua_is_mobile': parsed.is_mobile,
                    'ua_is_tablet': parsed.is_tablet,
                    'ua_is_pc': parsed.is_pc,
                    'ua_is_bot': parsed.is_bot
                }
            except Exception:
                return {
                    'ua_browser_family': None,
                    'ua_browser_version': None,
                    'ua_os_family': None,
                    'ua_os_version': None,
                    'ua_device_family': None,
                    'ua_device_brand': None,
                    'ua_is_mobile': False,
                    'ua_is_tablet': False,
                    'ua_is_pc': False,
                    'ua_is_bot': False
                }
        
        # Extract user agent features
        ua_features = df['userAgent'].apply(parse_ua)
        ua_df = pd.DataFrame(ua_features.tolist())
        
        # Optimize data types
        categorical_cols = ['ua_browser_family', 'ua_browser_version', 'ua_os_family', 
                           'ua_os_version', 'ua_device_family', 'ua_device_brand']
        for col in categorical_cols:
            ua_df[col] = ua_df[col].astype('category')
        
        return pd.concat([df, ua_df], axis=1)
    
    def _extract_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract temporal features from datetime column."""
        logger.info("Extracting temporal features...")
        
        # Ensure date column is datetime
        if not pd.api.types.is_datetime64_any_dtype(df['date']):
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
        # Extract temporal components
        df['hour'] = df['date'].dt.hour.astype('int8')
        df['day_of_week'] = df['date'].dt.dayofweek.astype('int8')
        df['day_of_month'] = df['date'].dt.day.astype('int8')
        df['month'] = df['date'].dt.month.astype('int8')
        df['quarter'] = df['date'].dt.quarter.astype('int8')
        df['year'] = df['date'].dt.year.astype('int16')
        
        # Business vs weekend
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(bool)
        
        # Time segments
        df['time_segment'] = pd.cut(df['hour'], 
                                   bins=[0, 6, 12, 18, 24], 
                                   labels=['night', 'morning', 'afternoon', 'evening'],
                                   include_lowest=True).astype('category')
        
        # Peak hours (typical business hours)
        df['is_peak_hour'] = ((df['hour'] >= 9) & (df['hour'] <= 17) & ~df['is_weekend']).astype(bool)
        
        return df
    
    def _extract_ip_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract features from IP addresses."""
        logger.info("Extracting IP features...")
        
        def classify_ip_type(ip):
            if pd.isna(ip):
                return 'unknown'
            
            try:
                # Simple IPv4 vs IPv6 classification
                if ':' in ip:
                    return 'ipv6'
                elif '.' in ip:
                    return 'ipv4'
                else:
                    return 'unknown'
            except:
                return 'unknown'
        
        df['ip_type'] = df['ip'].apply(classify_ip_type).astype('category')
        
        # Aggregate suspicious IP flags into a risk score
        suspicious_flags = ['isIpDatacenter', 'isIpAnonymous', 'isIpCrawler', 
                           'isIpPublicProxy', 'isIpVPN', 'isIpTOR', 'isIpResidentialProxy']
        
        existing_flags = [flag for flag in suspicious_flags if flag in df.columns]
        df['ip_risk_score'] = df[existing_flags].sum(axis=1).astype('int8')
        df['is_high_risk_ip'] = (df['ip_risk_score'] >= 2).astype(bool)
        
        return df


class TemporalAnalyzer:
    """Advanced temporal pattern analysis for traffic data."""
    
    @staticmethod
    @timing_decorator
    def analyze_traffic_patterns(df: pd.DataFrame, group_by: str = 'channelId') -> Dict[str, pd.DataFrame]:
        """Analyze traffic patterns across different time dimensions."""
        logger.info(f"Analyzing traffic patterns grouped by {group_by}...")
        
        results = {}
        
        # Hourly patterns
        hourly_traffic = df.groupby([group_by, 'hour']).size().reset_index(name='traffic_count')
        hourly_pivot = hourly_traffic.pivot(index=group_by, columns='hour', values='traffic_count').fillna(0)
        results['hourly_patterns'] = hourly_pivot
        
        # Daily patterns
        daily_traffic = df.groupby([group_by, 'day_of_week']).size().reset_index(name='traffic_count')
        daily_pivot = daily_traffic.pivot(index=group_by, columns='day_of_week', values='traffic_count').fillna(0)
        results['daily_patterns'] = daily_pivot
        
        # Time segment analysis
        if 'time_segment' in df.columns:
            segment_traffic = df.groupby([group_by, 'time_segment']).size().reset_index(name='traffic_count')
            segment_pivot = segment_traffic.pivot(index=group_by, columns='time_segment', values='traffic_count').fillna(0)
            results['time_segment_patterns'] = segment_pivot
        
        return results
    
    @staticmethod
    def detect_temporal_anomalies(df: pd.DataFrame, threshold: float = 2.0) -> pd.DataFrame:
        """Detect temporal anomalies using statistical methods."""
        logger.info("Detecting temporal anomalies...")
        
        # Group by hour and calculate traffic volume
        hourly_traffic = df.groupby('hour').size().reset_index(name='traffic_count')
        
        # Calculate z-scores
        hourly_traffic['z_score'] = np.abs(zscore(hourly_traffic['traffic_count']))
        hourly_traffic['is_anomaly'] = hourly_traffic['z_score'] > threshold
        
        return hourly_traffic


class ChannelSimilarityAnalyzer:
    """Channel similarity calculations using various metrics and features."""
    
    def __init__(self):
        self.scaler = StandardScaler()
    
    @timing_decorator
    def calculate_channel_similarity(self, df: pd.DataFrame, features: List[str] = None) -> pd.DataFrame:
        """Calculate similarity between channels using multiple metrics."""
        logger.info("Calculating channel similarity...")
        
        if features is None:
            # Default features for similarity calculation
            features = self._get_default_similarity_features(df)
        
        # Create channel profiles
        channel_profiles = self._create_channel_profiles(df, features)
        
        # Calculate similarity matrix
        similarity_matrix = self._calculate_cosine_similarity(channel_profiles)
        
        return similarity_matrix
    
    def _get_default_similarity_features(self, df: pd.DataFrame) -> List[str]:
        """Get default features for similarity calculation."""
        base_features = []
        
        # Traffic volume features
        if 'hour' in df.columns:
            base_features.extend([f'hour_{i}_traffic' for i in range(24)])
        
        # Device and browser features
        if 'device' in df.columns:
            base_features.append('mobile_ratio')
        
        if 'country' in df.columns:
            base_features.append('top_countries_diversity')
        
        if 'is_weekend' in df.columns:
            base_features.append('weekend_ratio')
        
        if 'ip_risk_score' in df.columns:
            base_features.append('avg_ip_risk_score')
        
        return base_features
    
    def _create_channel_profiles(self, df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
        """Create comprehensive profiles for each channel."""
        logger.info("Creating channel profiles...")
        
        profiles = []
        
        for channel_id in df['channelId'].unique():
            channel_data = df[df['channelId'] == channel_id]
            profile = {'channelId': channel_id}
            
            # Traffic volume by hour
            for hour in range(24):
                hour_traffic = len(channel_data[channel_data.get('hour', -1) == hour])
                profile[f'hour_{hour}_traffic'] = hour_traffic
            
            # Device distribution
            if 'device' in df.columns:
                total_traffic = len(channel_data)
                mobile_traffic = len(channel_data[channel_data['device'] == 'mobile'])
                profile['mobile_ratio'] = mobile_traffic / total_traffic if total_traffic > 0 else 0
            
            # Geographic diversity
            if 'country' in df.columns:
                unique_countries = channel_data['country'].nunique()
                profile['top_countries_diversity'] = unique_countries
            
            # Temporal patterns
            if 'is_weekend' in df.columns:
                weekend_traffic = len(channel_data[channel_data['is_weekend'] == True])
                total_traffic = len(channel_data)
                profile['weekend_ratio'] = weekend_traffic / total_traffic if total_traffic > 0 else 0
            
            # Risk metrics
            if 'ip_risk_score' in df.columns:
                profile['avg_ip_risk_score'] = channel_data['ip_risk_score'].mean()
            
            # Bot detection
            if 'isLikelyBot' in df.columns:
                bot_traffic = len(channel_data[channel_data['isLikelyBot'] == True])
                total_traffic = len(channel_data)
                profile['bot_ratio'] = bot_traffic / total_traffic if total_traffic > 0 else 0
            
            profiles.append(profile)
        
        profiles_df = pd.DataFrame(profiles)
        profiles_df = profiles_df.fillna(0)
        
        return profiles_df
    
    def _calculate_cosine_similarity(self, profiles_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate cosine similarity between channel profiles."""
        # Extract feature columns (exclude channelId)
        feature_cols = [col for col in profiles_df.columns if col != 'channelId']
        feature_matrix = profiles_df[feature_cols].values
        
        # Normalize features
        feature_matrix_scaled = self.scaler.fit_transform(feature_matrix)
        
        # Calculate cosine similarity
        n_channels = len(profiles_df)
        similarity_matrix = np.zeros((n_channels, n_channels))
        
        for i in range(n_channels):
            for j in range(n_channels):
                if i != j:
                    similarity = 1 - cosine(feature_matrix_scaled[i], feature_matrix_scaled[j])
                    similarity_matrix[i][j] = similarity
                else:
                    similarity_matrix[i][j] = 1.0
        
        # Create DataFrame with channel IDs as index and columns
        similarity_df = pd.DataFrame(
            similarity_matrix,
            index=profiles_df['channelId'],
            columns=profiles_df['channelId']
        )
        
        return similarity_df
    
    @timing_decorator
    def find_similar_channels(self, similarity_matrix: pd.DataFrame, 
                            channel_id: str, top_k: int = 10) -> pd.Series:
        """Find the most similar channels to a given channel."""
        if channel_id not in similarity_matrix.index:
            raise ValueError(f"Channel {channel_id} not found in similarity matrix")
        
        similarities = similarity_matrix.loc[channel_id]
        # Exclude self-similarity
        similarities = similarities[similarities.index != channel_id]
        
        return similarities.nlargest(top_k)


class TrafficPatternAnalyzer:
    """Comprehensive traffic pattern analysis for behavioral insights."""
    
    @staticmethod
    @timing_decorator
    def analyze_device_patterns(df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze device usage patterns across channels."""
        logger.info("Analyzing device patterns...")
        
        results = {}
        
        # Overall device distribution
        device_dist = df['device'].value_counts(normalize=True)
        results['device_distribution'] = device_dist
        
        # Device patterns by channel
        if 'channelId' in df.columns:
            channel_device = df.groupby(['channelId', 'device']).size().unstack(fill_value=0)
            channel_device_pct = channel_device.div(channel_device.sum(axis=1), axis=0)
            results['channel_device_patterns'] = channel_device_pct
        
        # Device patterns by time
        if 'hour' in df.columns:
            hourly_device = df.groupby(['hour', 'device']).size().unstack(fill_value=0)
            hourly_device_pct = hourly_device.div(hourly_device.sum(axis=1), axis=0)
            results['hourly_device_patterns'] = hourly_device_pct
        
        return results
    
    @staticmethod
    @timing_decorator
    def analyze_geographic_patterns(df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze geographic distribution and patterns."""
        logger.info("Analyzing geographic patterns...")
        
        results = {}
        
        # Country distribution
        country_dist = df['country'].value_counts()
        results['country_distribution'] = country_dist
        
        # City analysis (if available)
        if 'city_name' in df.columns:
            city_dist = df['city_name'].value_counts().head(50)  # Top 50 cities
            results['city_distribution'] = city_dist
        
        # Timezone analysis
        if 'timezone' in df.columns:
            timezone_dist = df['timezone'].value_counts()
            results['timezone_distribution'] = timezone_dist
        
        # Channel geographic diversity
        if 'channelId' in df.columns:
            channel_geo_diversity = df.groupby('channelId')['country'].nunique().sort_values(ascending=False)
            results['channel_geographic_diversity'] = channel_geo_diversity
        
        return results
    
    @staticmethod
    @timing_decorator
    def analyze_browser_patterns(df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze browser usage patterns."""
        logger.info("Analyzing browser patterns...")
        
        results = {}
        
        # Browser distribution
        browser_dist = df['browser'].value_counts()
        results['browser_distribution'] = browser_dist
        
        # Browser version analysis
        if 'browserMajorVersion' in df.columns:
            version_dist = df.groupby(['browser', 'browserMajorVersion']).size().reset_index(name='count')
            results['browser_version_distribution'] = version_dist
        
        # User agent analysis (if extracted)
        if 'ua_browser_family' in df.columns:
            ua_browser_dist = df['ua_browser_family'].value_counts()
            results['ua_browser_distribution'] = ua_browser_dist
        
        # OS analysis
        if 'os_name' in df.columns:
            os_dist = df['os_name'].value_counts()
            results['os_distribution'] = os_dist
        
        return results


class QualityScoreCalculator:
    """Quality scoring and anomaly detection for traffic analysis."""
    
    def __init__(self):
        self.risk_weights = {
            'bot_ratio': 0.3,
            'high_risk_ip_ratio': 0.25,
            'datacenter_ratio': 0.2,
            'anonymization_ratio': 0.15,
            'temporal_anomaly_score': 0.1
        }
    
    @timing_decorator
    def calculate_channel_quality_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive quality scores for each channel."""
        logger.info("Calculating channel quality scores...")
        
        quality_scores = []
        
        for channel_id in df['channelId'].unique():
            channel_data = df[df['channelId'] == channel_id]
            score_components = self._calculate_score_components(channel_data)
            
            # Calculate weighted quality score (0-100, higher is better)
            quality_score = 100 - (
                score_components['bot_ratio'] * self.risk_weights['bot_ratio'] * 100 +
                score_components['high_risk_ip_ratio'] * self.risk_weights['high_risk_ip_ratio'] * 100 +
                score_components['datacenter_ratio'] * self.risk_weights['datacenter_ratio'] * 100 +
                score_components['anonymization_ratio'] * self.risk_weights['anonymization_ratio'] * 100 +
                score_components['temporal_anomaly_score'] * self.risk_weights['temporal_anomaly_score'] * 100
            )
            
            quality_scores.append({
                'channelId': channel_id,
                'quality_score': max(0, quality_score),  # Ensure non-negative
                'traffic_volume': len(channel_data),
                **score_components
            })
        
        return pd.DataFrame(quality_scores)
    
    def _calculate_score_components(self, channel_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate individual components of the quality score."""
        total_traffic = len(channel_data)
        
        components = {}
        
        # Bot ratio
        if 'isLikelyBot' in channel_data.columns:
            bot_count = channel_data['isLikelyBot'].sum()
            components['bot_ratio'] = bot_count / total_traffic if total_traffic > 0 else 0
        else:
            components['bot_ratio'] = 0
        
        # High risk IP ratio
        if 'is_high_risk_ip' in channel_data.columns:
            high_risk_count = channel_data['is_high_risk_ip'].sum()
            components['high_risk_ip_ratio'] = high_risk_count / total_traffic if total_traffic > 0 else 0
        else:
            components['high_risk_ip_ratio'] = 0
        
        # Datacenter ratio
        if 'isIpDatacenter' in channel_data.columns:
            datacenter_count = channel_data['isIpDatacenter'].sum()
            components['datacenter_ratio'] = datacenter_count / total_traffic if total_traffic > 0 else 0
        else:
            components['datacenter_ratio'] = 0
        
        # Anonymization tools ratio
        anonymization_flags = ['isIpAnonymous', 'isIpVPN', 'isIpTOR', 'isIpPublicProxy']
        anonymization_count = 0
        for flag in anonymization_flags:
            if flag in channel_data.columns:
                anonymization_count += channel_data[flag].sum()
        
        components['anonymization_ratio'] = anonymization_count / total_traffic if total_traffic > 0 else 0
        
        # Temporal anomaly score (simplified)
        if 'hour' in channel_data.columns:
            hourly_dist = channel_data['hour'].value_counts(normalize=True)
            # Calculate standard deviation of hourly distribution
            components['temporal_anomaly_score'] = hourly_dist.std()
        else:
            components['temporal_anomaly_score'] = 0
        
        return components
    
    @timing_decorator
    def detect_anomalous_channels(self, quality_scores_df: pd.DataFrame, 
                                threshold: float = 30.0) -> pd.DataFrame:
        """Detect channels with anomalously low quality scores."""
        logger.info(f"Detecting anomalous channels with quality score < {threshold}")
        
        anomalous = quality_scores_df[quality_scores_df['quality_score'] < threshold].copy()
        anomalous = anomalous.sort_values('quality_score')
        
        return anomalous
    
    @timing_decorator
    def cluster_channels_by_quality(self, quality_scores_df: pd.DataFrame, 
                                  n_clusters: int = 5) -> pd.DataFrame:
        """Cluster channels based on quality metrics using DBSCAN."""
        logger.info("Clustering channels by quality metrics...")
        
        # Select features for clustering
        feature_cols = ['bot_ratio', 'high_risk_ip_ratio', 'datacenter_ratio', 
                       'anonymization_ratio', 'temporal_anomaly_score']
        
        existing_features = [col for col in feature_cols if col in quality_scores_df.columns]
        
        if len(existing_features) < 2:
            logger.warning("Insufficient features for clustering")
            quality_scores_df['cluster'] = 0
            return quality_scores_df
        
        # Prepare data for clustering
        feature_data = quality_scores_df[existing_features].fillna(0)
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(feature_data)
        
        # Apply DBSCAN clustering
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        clusters = dbscan.fit_predict(scaled_features)
        
        quality_scores_df = quality_scores_df.copy()
        quality_scores_df['cluster'] = clusters
        
        logger.info(f"Found {len(set(clusters))} clusters (including noise as -1)")
        
        return quality_scores_df


class PerformanceBenchmark:
    """Performance benchmarking and optimization testing."""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.results = {}
    
    @timing_decorator
    def benchmark_data_loading(self, nrows: int = 100000):
        """Benchmark different data loading approaches."""
        logger.info("Benchmarking data loading approaches...")
        
        results = {}
        
        # Test pandas loading
        start_time = time.perf_counter()
        loader_pandas = TrafficDataLoader(DataLoadConfig(use_polars=False))
        df_pandas = loader_pandas.load_data(self.file_path, nrows=nrows)
        pandas_time = time.perf_counter() - start_time
        pandas_memory = df_pandas.memory_usage(deep=True).sum() / 1024**2
        
        results['pandas'] = {
            'load_time': pandas_time,
            'memory_mb': pandas_memory,
            'records': len(df_pandas)
        }
        
        # Test polars loading
        start_time = time.perf_counter()
        loader_polars = TrafficDataLoader(DataLoadConfig(use_polars=True))
        df_polars = loader_polars.load_data(self.file_path, nrows=nrows)
        polars_time = time.perf_counter() - start_time
        
        results['polars'] = {
            'load_time': polars_time,
            'memory_mb': 'N/A',  # Polars memory usage is different
            'records': len(df_polars)
        }
        
        self.results['data_loading'] = results
        return results
    
    @timing_decorator
    def benchmark_feature_extraction(self, nrows: int = 50000):
        """Benchmark feature extraction performance."""
        logger.info("Benchmarking feature extraction...")
        
        loader = TrafficDataLoader()
        df = loader.load_data(self.file_path, nrows=nrows)
        
        extractor = AdvancedFeatureExtractor()
        
        start_time = time.perf_counter()
        df_enriched = extractor.extract_all_features(df)
        extraction_time = time.perf_counter() - start_time
        
        results = {
            'extraction_time': extraction_time,
            'original_features': len(df.columns),
            'extracted_features': len(df_enriched.columns),
            'records_processed': len(df),
            'features_per_second': (len(df_enriched.columns) - len(df.columns)) * len(df) / extraction_time
        }
        
        self.results['feature_extraction'] = results
        return results
    
    @timing_decorator
    def benchmark_analysis_operations(self, nrows: int = 50000):
        """Benchmark various analysis operations."""
        logger.info("Benchmarking analysis operations...")
        
        # Load and prepare data
        loader = TrafficDataLoader()
        extractor = AdvancedFeatureExtractor()
        df = loader.load_data(self.file_path, nrows=nrows)
        df_enriched = extractor.extract_all_features(df)
        
        results = {}
        
        # Channel similarity benchmark
        similarity_analyzer = ChannelSimilarityAnalyzer()
        start_time = time.perf_counter()
        similarity_matrix = similarity_analyzer.calculate_channel_similarity(df_enriched)
        similarity_time = time.perf_counter() - start_time
        
        results['channel_similarity'] = {
            'time': similarity_time,
            'channels_analyzed': len(similarity_matrix),
            'matrix_size': similarity_matrix.shape
        }
        
        # Traffic pattern analysis benchmark
        pattern_analyzer = TrafficPatternAnalyzer()
        start_time = time.perf_counter()
        device_patterns = pattern_analyzer.analyze_device_patterns(df_enriched)
        geo_patterns = pattern_analyzer.analyze_geographic_patterns(df_enriched)
        browser_patterns = pattern_analyzer.analyze_browser_patterns(df_enriched)
        pattern_time = time.perf_counter() - start_time
        
        results['pattern_analysis'] = {
            'time': pattern_time,
            'device_patterns': len(device_patterns),
            'geo_patterns': len(geo_patterns),
            'browser_patterns': len(browser_patterns)
        }
        
        # Quality scoring benchmark
        quality_calculator = QualityScoreCalculator()
        start_time = time.perf_counter()
        quality_scores = quality_calculator.calculate_channel_quality_scores(df_enriched)
        quality_time = time.perf_counter() - start_time
        
        results['quality_scoring'] = {
            'time': quality_time,
            'channels_scored': len(quality_scores),
            'avg_quality_score': quality_scores['quality_score'].mean()
        }
        
        self.results['analysis_operations'] = results
        return results
    
    def generate_performance_report(self) -> str:
        """Generate a comprehensive performance report."""
        if not self.results:
            return "No benchmark results available. Run benchmarks first."
        
        report = ["=" * 60]
        report.append("TRAFFIC ANALYZER PERFORMANCE REPORT")
        report.append("=" * 60)
        
        for category, results in self.results.items():
            report.append(f"\n{category.upper().replace('_', ' ')}:")
            report.append("-" * 40)
            
            if isinstance(results, dict):
                for key, value in results.items():
                    if isinstance(value, dict):
                        report.append(f"  {key}:")
                        for sub_key, sub_value in value.items():
                            report.append(f"    {sub_key}: {sub_value}")
                    else:
                        report.append(f"  {key}: {value}")
        
        return "\n".join(report)


class TrafficAnalyzerPipeline:
    """Complete pipeline for traffic data analysis."""
    
    def __init__(self, file_path: str, config: Dict[str, Any] = None):
        self.file_path = file_path
        self.config = config or {}
        
        # Initialize components
        self.loader = TrafficDataLoader(self.config.get('data_load_config'))
        self.extractor = AdvancedFeatureExtractor(self.config.get('feature_config'))
        self.temporal_analyzer = TemporalAnalyzer()
        self.similarity_analyzer = ChannelSimilarityAnalyzer()
        self.pattern_analyzer = TrafficPatternAnalyzer()
        self.quality_calculator = QualityScoreCalculator()
        
        # Results storage
        self.raw_data = None
        self.enriched_data = None
        self.analysis_results = {}
    
    @timing_decorator
    def run_full_analysis(self, nrows: Optional[int] = None, use_chunked: bool = False):
        """Run the complete traffic analysis pipeline."""
        logger.info("Starting full traffic analysis pipeline...")
        
        if use_chunked:
            self._run_chunked_analysis()
        else:
            self._run_batch_analysis(nrows)
        
        logger.info("Full analysis pipeline completed successfully")
    
    def _run_batch_analysis(self, nrows: Optional[int] = None):
        """Run analysis on full dataset or subset."""
        # Load data
        logger.info("Loading data...")
        self.raw_data = self.loader.load_data(self.file_path, nrows=nrows)
        
        # Extract features
        logger.info("Extracting features...")
        self.enriched_data = self.extractor.extract_all_features(self.raw_data)
        
        # Run analyses
        self._run_all_analyses()
    
    def _run_chunked_analysis(self):
        """Run analysis using chunked processing for large datasets."""
        logger.info("Running chunked analysis...")
        
        all_quality_scores = []
        all_similarity_results = []
        combined_patterns = {}
        
        chunk_count = 0
        for chunk in self.loader.load_data_chunked(self.file_path):
            chunk_count += 1
            logger.info(f"Processing chunk {chunk_count}...")
            
            # Extract features for chunk
            enriched_chunk = self.extractor.extract_all_features(chunk)
            
            # Calculate quality scores for chunk
            chunk_quality = self.quality_calculator.calculate_channel_quality_scores(enriched_chunk)
            all_quality_scores.append(chunk_quality)
            
            # Accumulate pattern data
            chunk_patterns = self.pattern_analyzer.analyze_device_patterns(enriched_chunk)
            if not combined_patterns:
                combined_patterns = chunk_patterns
            else:
                # Combine patterns across chunks (simplified)
                for key, value in chunk_patterns.items():
                    if key in combined_patterns:
                        if hasattr(value, 'add'):
                            combined_patterns[key] = combined_patterns[key].add(value, fill_value=0)
        
        # Combine results from all chunks
        if all_quality_scores:
            self.analysis_results['quality_scores'] = pd.concat(all_quality_scores, ignore_index=True)
        
        self.analysis_results['pattern_analysis'] = combined_patterns
        
        logger.info(f"Processed {chunk_count} chunks successfully")
    
    def _run_all_analyses(self):
        """Run all analysis components on enriched data."""
        logger.info("Running comprehensive analysis...")
        
        # Temporal analysis
        self.analysis_results['temporal_patterns'] = self.temporal_analyzer.analyze_traffic_patterns(self.enriched_data)
        
        # Channel similarity
        self.analysis_results['channel_similarity'] = self.similarity_analyzer.calculate_channel_similarity(self.enriched_data)
        
        # Pattern analysis
        self.analysis_results['device_patterns'] = self.pattern_analyzer.analyze_device_patterns(self.enriched_data)
        self.analysis_results['geographic_patterns'] = self.pattern_analyzer.analyze_geographic_patterns(self.enriched_data)
        self.analysis_results['browser_patterns'] = self.pattern_analyzer.analyze_browser_patterns(self.enriched_data)
        
        # Quality scoring
        self.analysis_results['quality_scores'] = self.quality_calculator.calculate_channel_quality_scores(self.enriched_data)
        self.analysis_results['anomalous_channels'] = self.quality_calculator.detect_anomalous_channels(
            self.analysis_results['quality_scores']
        )
        self.analysis_results['quality_clusters'] = self.quality_calculator.cluster_channels_by_quality(
            self.analysis_results['quality_scores']
        )
    
    def get_summary_report(self) -> str:
        """Generate a summary report of all analysis results."""
        if not self.analysis_results:
            return "No analysis results available. Run analysis first."
        
        report = ["=" * 60]
        report.append("TRAFFIC ANALYSIS SUMMARY REPORT")
        report.append("=" * 60)
        
        if self.enriched_data is not None:
            report.append(f"\nDataset Overview:")
            report.append(f"  Total Records: {len(self.enriched_data):,}")
            report.append(f"  Total Features: {len(self.enriched_data.columns)}")
            report.append(f"  Date Range: {self.enriched_data['date'].min()} to {self.enriched_data['date'].max()}")
            report.append(f"  Unique Channels: {self.enriched_data['channelId'].nunique()}")
            report.append(f"  Unique Countries: {self.enriched_data['country'].nunique()}")
        
        if 'quality_scores' in self.analysis_results:
            quality_df = self.analysis_results['quality_scores']
            report.append(f"\nQuality Analysis:")
            report.append(f"  Average Quality Score: {quality_df['quality_score'].mean():.2f}")
            report.append(f"  Lowest Quality Score: {quality_df['quality_score'].min():.2f}")
            report.append(f"  Highest Quality Score: {quality_df['quality_score'].max():.2f}")
            
            if 'anomalous_channels' in self.analysis_results:
                anomalous_count = len(self.analysis_results['anomalous_channels'])
                report.append(f"  Anomalous Channels: {anomalous_count}")
        
        if 'channel_similarity' in self.analysis_results:
            similarity_matrix = self.analysis_results['channel_similarity']
            report.append(f"\nChannel Similarity:")
            report.append(f"  Channels Analyzed: {len(similarity_matrix)}")
            report.append(f"  Average Similarity: {similarity_matrix.values.mean():.3f}")
        
        return "\n".join(report)


if __name__ == "__main__":
    # Comprehensive testing and demonstration
    file_path = "/home/fiod/shimshi/bq-results-20250804-141411-1754316868932.csv"
    
    print("=" * 60)
    print("TRAFFIC ANALYZER - COMPREHENSIVE TESTING")
    print("=" * 60)
    
    # Test 1: Basic functionality with sample data
    print("\n1. Testing basic functionality with 5000 records...")
    pipeline = TrafficAnalyzerPipeline(file_path)
    pipeline.run_full_analysis(nrows=5000)
    print(pipeline.get_summary_report())
    
    # Test 2: Performance benchmarking
    print("\n2. Running performance benchmarks...")
    benchmark = PerformanceBenchmark(file_path)
    benchmark.benchmark_data_loading(nrows=10000)
    benchmark.benchmark_feature_extraction(nrows=5000)
    benchmark.benchmark_analysis_operations(nrows=5000)
    print(benchmark.generate_performance_report())
    
    # Test 3: Chunked processing demonstration
    print("\n3. Testing chunked processing (2 chunks)...")
    loader = TrafficDataLoader()
    chunk_count = 0
    total_records = 0
    
    for chunk in loader.load_data_chunked(file_path):
        chunk_count += 1
        total_records += len(chunk)
        print(f"Processed chunk {chunk_count}: {len(chunk):,} records")
        
        if chunk_count >= 2:  # Limit for testing
            break
    
    print(f"Total records processed in chunks: {total_records:,}")
    
    print("\n" + "=" * 60)
    print("TESTING COMPLETED SUCCESSFULLY")
    print("=" * 60)