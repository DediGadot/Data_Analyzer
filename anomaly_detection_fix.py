"""
Quick fix for anomaly detection channelId issues
This patches the anomaly detection methods to handle DataFrame index/column issues properly.
"""

import pandas as pd
import numpy as np

def fix_channelid_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Comprehensive fix for channelId index/column issues.
    
    Args:
        df: DataFrame that may have channelId issues
        
    Returns:
        DataFrame with channelId properly as a column
    """
    df = df.copy()
    
    # Case 1: channelId is both index and column - remove from index, keep column
    if 'channelId' in df.columns and df.index.name == 'channelId':
        df = df.reset_index(drop=True)
    
    # Case 2: channelId is in MultiIndex - flatten and keep as column
    elif isinstance(df.index, pd.MultiIndex) and 'channelId' in df.index.names:
        df = df.reset_index()
    
    # Case 3: channelId is only in index - move to column
    elif df.index.name == 'channelId' and 'channelId' not in df.columns:
        df = df.reset_index()
    
    # Case 4: No channelId anywhere - this is an error
    elif 'channelId' not in df.columns:
        raise ValueError(f"No channelId found. Columns: {list(df.columns)}, Index: {df.index.name}")
    
    return df

def patch_anomaly_detection():
    """Patch the anomaly detection module to handle channelId issues."""
    import anomaly_detection
    
    # Store original methods
    original_temporal = anomaly_detection.AnomalyDetector.detect_temporal_anomalies
    original_geographic = anomaly_detection.AnomalyDetector.detect_geographic_anomalies
    original_device = anomaly_detection.AnomalyDetector.detect_device_anomalies
    original_behavioral = anomaly_detection.AnomalyDetector.detect_behavioral_anomalies
    original_volume = anomaly_detection.AnomalyDetector.detect_volume_anomalies
    
    def patched_temporal_anomalies(self, df):
        try:
            df = fix_channelid_dataframe(df)
            return original_temporal(self, df)
        except Exception as e:
            print(f"Temporal anomaly detection failed: {e}")
            return pd.DataFrame()
    
    def patched_geographic_anomalies(self, df):
        try:
            df = fix_channelid_dataframe(df)
            return original_geographic(self, df)
        except Exception as e:
            print(f"Geographic anomaly detection failed: {e}")
            return pd.DataFrame()
    
    def patched_device_anomalies(self, df):
        try:
            df = fix_channelid_dataframe(df)
            return original_device(self, df)
        except Exception as e:
            print(f"Device anomaly detection failed: {e}")
            return pd.DataFrame()
    
    def patched_behavioral_anomalies(self, df):
        try:
            df = fix_channelid_dataframe(df)
            return original_behavioral(self, df)
        except Exception as e:
            print(f"Behavioral anomaly detection failed: {e}")
            return pd.DataFrame()
    
    def patched_volume_anomalies(self, df):
        try:
            df = fix_channelid_dataframe(df)
            return original_volume(self, df)
        except Exception as e:
            print(f"Volume anomaly detection failed: {e}")
            return pd.DataFrame()
    
    # Apply patches
    anomaly_detection.AnomalyDetector.detect_temporal_anomalies = patched_temporal_anomalies
    anomaly_detection.AnomalyDetector.detect_geographic_anomalies = patched_geographic_anomalies
    anomaly_detection.AnomalyDetector.detect_device_anomalies = patched_device_anomalies
    anomaly_detection.AnomalyDetector.detect_behavioral_anomalies = patched_behavioral_anomalies
    anomaly_detection.AnomalyDetector.detect_volume_anomalies = patched_volume_anomalies
    
    print("Anomaly detection methods patched successfully!")

if __name__ == "__main__":
    patch_anomaly_detection()