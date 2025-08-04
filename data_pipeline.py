"""
Data Pipeline for Fraud Detection ML Models
Efficiently processes large-scale advertising traffic data for fraud detection and quality scoring.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
import sqlite3
from datetime import datetime, timedelta
import multiprocessing as mp
from functools import partial
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataPipeline:
    """
    Scalable data pipeline for processing advertising traffic data.
    Handles large datasets with memory-efficient chunking and parallel processing.
    """
    
    def __init__(self, data_path: str, chunk_size: int = 10000):
        self.data_path = data_path
        self.chunk_size = chunk_size
        self.schema = self._get_schema()
        
    def _get_schema(self) -> Dict:
        """Define expected data schema and types."""
        return {
            'date': 'datetime64[ns]',
            'keyword': 'string',
            'country': 'category',
            'browser': 'category',
            'device': 'category',
            'referrer': 'string',
            'ip': 'string',
            'publisherId': 'string',
            'channelId': 'string',
            'advertiserId': 'string',
            'feedId': 'string',
            'browserMajorVersion': 'float32',
            'userId': 'string',
            'isLikelyBot': 'boolean',
            'ipClassification': 'category',
            'isIpDatacenter': 'boolean',
            'datacenterName': 'string',
            'ipHostName': 'string',
            'isIpAnonymous': 'boolean',
            'isIpCrawler': 'boolean',
            'isIpPublicProxy': 'boolean',
            'isIpVPN': 'boolean',
            'isIpHostingService': 'boolean',
            'isIpTOR': 'boolean',
            'isIpResidentialProxy': 'boolean',
            'performance': 'string',
            'detection': 'string',
            'platform': 'category',
            'location': 'string',
            'userAgent': 'string'
        }
    
    def load_data_chunked(self, sample_fraction: Optional[float] = None) -> pd.DataFrame:
        """
        Load data in chunks for memory efficiency.
        
        Args:
            sample_fraction: Optional fraction of data to sample (for development)
            
        Returns:
            Combined DataFrame with optimized dtypes
        """
        logger.info(f"Loading data from {self.data_path}")
        
        chunks = []
        total_rows = 0
        
        try:
            # Read in chunks to manage memory
            chunk_iter = pd.read_csv(
                self.data_path,
                chunksize=self.chunk_size,
                dtype={k: v for k, v in self.schema.items() if v != 'datetime64[ns]'},
                parse_dates=['date'],
                low_memory=False
            )
            
            for i, chunk in enumerate(chunk_iter):
                # Apply sampling if specified
                if sample_fraction and sample_fraction < 1.0:
                    chunk = chunk.sample(frac=sample_fraction, random_state=42)
                
                # Basic cleaning
                chunk = self._clean_chunk(chunk)
                chunks.append(chunk)
                total_rows += len(chunk)
                
                if i % 10 == 0:
                    logger.info(f"Processed {i+1} chunks, {total_rows:,} rows")
                    
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
        
        # Combine chunks
        df = pd.concat(chunks, ignore_index=True)
        logger.info(f"Loaded {len(df):,} total rows")
        
        # Optimize memory usage
        df = self._optimize_dtypes(df)
        
        return df
    
    def _clean_chunk(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean individual chunk of data."""
        # Remove duplicates within chunk
        df = df.drop_duplicates()
        
        # Handle missing values
        boolean_columns = [col for col, dtype in self.schema.items() 
                          if dtype == 'boolean' and col in df.columns]
        df[boolean_columns] = df[boolean_columns].fillna(False)
        
        # Clean IP addresses (basic validation)
        if 'ip' in df.columns:
            df['ip'] = df['ip'].astype(str)
            # Remove obviously invalid IPs
            df = df[~df['ip'].str.contains('nan|None|null', case=False, na=False)]
        
        return df
    
    def _optimize_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame memory usage."""
        logger.info("Optimizing data types for memory efficiency")
        
        # Convert object columns with low cardinality to category
        for col in df.select_dtypes(include=['object']).columns:
            if col not in ['ip', 'userId', 'userAgent', 'keyword']:  # Keep these as strings
                unique_ratio = df[col].nunique() / len(df)
                if unique_ratio < 0.5:  # Less than 50% unique values
                    df[col] = df[col].astype('category')
        
        # Optimize numeric types
        for col in df.select_dtypes(include=['int64']).columns:
            df[col] = pd.to_numeric(df[col], downcast='integer')
            
        for col in df.select_dtypes(include=['float64']).columns:
            df[col] = pd.to_numeric(df[col], downcast='float')
        
        return df
    
    def create_sqlite_db(self, df: pd.DataFrame, db_path: str = "traffic_data.db"):
        """Create SQLite database for efficient querying."""
        logger.info(f"Creating SQLite database: {db_path}")
        
        conn = sqlite3.connect(db_path)
        
        # Write main table
        df.to_sql('traffic_data', conn, if_exists='replace', index=False, chunksize=10000)
        
        # Create indexes for common queries
        cursor = conn.cursor()
        indexes = [
            "CREATE INDEX idx_channel_id ON traffic_data(channelId)",
            "CREATE INDEX idx_publisher_id ON traffic_data(publisherId)",
            "CREATE INDEX idx_date ON traffic_data(date)",
            "CREATE INDEX idx_country ON traffic_data(country)",
            "CREATE INDEX idx_is_likely_bot ON traffic_data(isLikelyBot)",
            "CREATE INDEX idx_ip_classification ON traffic_data(ipClassification)"
        ]
        
        for idx in indexes:
            try:
                cursor.execute(idx)
            except sqlite3.OperationalError:
                pass  # Index might already exist
        
        conn.commit()
        conn.close()
        logger.info("SQLite database created successfully")
    
    def get_data_summary(self, df: pd.DataFrame) -> Dict:
        """Generate comprehensive data summary."""
        summary = {
            'total_records': len(df),
            'date_range': {
                'start': df['date'].min(),
                'end': df['date'].max()
            },
            'unique_counts': {
                'channels': df['channelId'].nunique(),
                'publishers': df['publisherId'].nunique(),
                'advertisers': df['advertiserId'].nunique(),
                'countries': df['country'].nunique(),
                'ips': df['ip'].nunique(),
                'users': df['userId'].nunique()
            },
            'fraud_indicators': {
                'likely_bots': df['isLikelyBot'].sum(),
                'datacenter_ips': df['isIpDatacenter'].sum() if 'isIpDatacenter' in df.columns else 0,
                'anonymous_ips': df['isIpAnonymous'].sum() if 'isIpAnonymous' in df.columns else 0,
                'vpn_ips': df['isIpVPN'].sum() if 'isIpVPN' in df.columns else 0,
                'tor_ips': df['isIpTOR'].sum() if 'isIpTOR' in df.columns else 0
            },
            'top_countries': df['country'].value_counts().head(10).to_dict(),
            'top_browsers': df['browser'].value_counts().head(5).to_dict(),
            'device_distribution': df['device'].value_counts().to_dict()
        }
        
        return summary
    
    def validate_data_quality(self, df: pd.DataFrame) -> Dict:
        """Validate data quality and identify issues."""
        quality_report = {
            'missing_values': df.isnull().sum().to_dict(),
            'duplicate_rows': df.duplicated().sum(),
            'invalid_dates': 0,
            'suspicious_patterns': {}
        }
        
        # Check for invalid dates
        if 'date' in df.columns:
            try:
                invalid_dates = pd.to_datetime(df['date'], errors='coerce').isnull().sum()
                quality_report['invalid_dates'] = invalid_dates
            except:
                pass
        
        # Check for suspicious patterns
        if 'ip' in df.columns:
            # Count requests per IP
            ip_counts = df['ip'].value_counts()
            quality_report['suspicious_patterns']['high_volume_ips'] = (ip_counts > 1000).sum()
        
        if 'userId' in df.columns:
            # Count requests per user
            user_counts = df['userId'].value_counts()
            quality_report['suspicious_patterns']['high_volume_users'] = (user_counts > 100).sum()
        
        return quality_report

def main():
    """Main execution function for data pipeline."""
    # Initialize pipeline
    pipeline = DataPipeline("/home/fiod/shimshi/bq-results-20250804-141411-1754316868932.csv")
    
    # Load data (sample for development)
    logger.info("Starting data pipeline...")
    df = pipeline.load_data_chunked(sample_fraction=0.1)  # 10% sample for development
    
    # Generate summary
    summary = pipeline.get_data_summary(df)
    logger.info(f"Data Summary: {summary}")
    
    # Validate data quality
    quality_report = pipeline.validate_data_quality(df)
    logger.info(f"Quality Report: {quality_report}")
    
    # Create SQLite database for efficient access
    pipeline.create_sqlite_db(df)
    
    return df, summary, quality_report

if __name__ == "__main__":
    df, summary, quality_report = main()