"""
Enhanced Fraud Classification Module
Provides row-level fraud classification with detailed scoring and reason codes
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class FraudReason(Enum):
    """Enumeration of fraud reason codes"""
    HIGH_BOT_ACTIVITY = "high_bot_activity"
    SUSPICIOUS_IP_PATTERN = "suspicious_ip_pattern"
    TEMPORAL_ANOMALY = "temporal_anomaly"
    GEOGRAPHIC_ANOMALY = "geographic_anomaly"
    DEVICE_ANOMALY = "device_anomaly"
    BEHAVIORAL_ANOMALY = "behavioral_anomaly"
    VOLUME_ANOMALY = "volume_anomaly"
    LOW_QUALITY_SCORE = "low_quality_score"
    MULTIPLE_INDICATORS = "multiple_indicators"
    DATACENTER_IP = "datacenter_ip"
    ANONYMOUS_IP = "anonymous_ip"
    CLEAN_PATTERN = "clean_pattern"
    INSUFFICIENT_DATA = "insufficient_data"

@dataclass
class ClassificationResult:
    """Result of fraud classification for a single row"""
    classification: str  # "good_account" or "fraud"
    quality_score: float  # 0-10 scale
    risk_score: float  # 0-1 probability
    confidence: float  # 0-1 confidence level
    reason_codes: List[str]  # List of reason codes
    temporal_anomaly: bool
    geographic_anomaly: bool
    device_anomaly: bool
    behavioral_anomaly: bool
    volume_anomaly: bool
    overall_anomaly_count: int

class FraudClassifier:
    """
    Enhanced fraud classifier that provides row-level fraud detection
    with detailed scoring and interpretable reason codes
    """
    
    def __init__(self, 
                 quality_threshold_low: float = 3.0,
                 quality_threshold_high: float = 7.0,
                 anomaly_threshold_high: int = 3,
                 risk_threshold: float = 0.5):
        """
        Initialize classifier with configurable thresholds
        
        Args:
            quality_threshold_low: Below this = likely fraud
            quality_threshold_high: Above this = likely good
            anomaly_threshold_high: Anomaly count threshold for fraud
            risk_threshold: Risk score threshold for classification
        """
        self.quality_threshold_low = quality_threshold_low
        self.quality_threshold_high = quality_threshold_high
        self.anomaly_threshold_high = anomaly_threshold_high
        self.risk_threshold = risk_threshold
        
        # Business rule weights for ensemble scoring
        self.weights = {
            'quality_score': 0.4,
            'anomaly_count': 0.3,
            'bot_indicator': 0.15,
            'ip_risk': 0.15
        }
        
        logger.info(f"FraudClassifier initialized with thresholds: quality_low={quality_threshold_low}, "
                   f"quality_high={quality_threshold_high}, anomaly_high={anomaly_threshold_high}, "
                   f"risk={risk_threshold}")
    
    def classify_dataset(self, 
                        original_df: pd.DataFrame,
                        quality_results: pd.DataFrame,
                        anomaly_results: pd.DataFrame,
                        features_df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Classify entire dataset with row-level results
        
        Args:
            original_df: Original input data with all columns
            quality_results: Quality scoring results (channel-level)
            anomaly_results: Anomaly detection results (row-level or channel-level)
            features_df: Optional engineered features
            
        Returns:
            DataFrame with original data + classification columns
        """
        logger.info(f"Classifying {len(original_df)} rows with fraud detection")
        
        # Start with original data
        result_df = original_df.copy()
        
        # Add index to preserve row mapping
        result_df['_original_index'] = result_df.index
        
        # Initialize classification columns with defaults
        result_df['classification'] = 'good_account'
        result_df['quality_score'] = 5.0
        result_df['risk_score'] = 0.0
        result_df['confidence'] = 0.5
        result_df['reason_codes'] = ''
        result_df['temporal_anomaly'] = False
        result_df['geographic_anomaly'] = False
        result_df['device_anomaly'] = False
        result_df['behavioral_anomaly'] = False
        result_df['volume_anomaly'] = False
        result_df['overall_anomaly_count'] = 0
        
        # Create channel-to-quality mapping
        quality_mapping = self._create_quality_mapping(quality_results)
        
        # Create anomaly mapping (handle both row-level and channel-level)
        anomaly_mapping = self._create_anomaly_mapping(anomaly_results, original_df)
        
        # Process each row for classification
        logger.info("Processing individual row classifications...")
        
        classifications = []
        batch_size = 10000
        
        for i in range(0, len(result_df), batch_size):
            batch = result_df.iloc[i:i+batch_size]
            batch_classifications = self._classify_batch(
                batch, quality_mapping, anomaly_mapping, features_df
            )
            classifications.extend(batch_classifications)
            
            if (i // batch_size + 1) % 10 == 0:
                logger.info(f"Processed {min(i + batch_size, len(result_df))} / {len(result_df)} rows")
        
        # Apply classifications to result DataFrame
        for i, classification in enumerate(classifications):
            result_df.at[i, 'classification'] = classification.classification
            result_df.at[i, 'quality_score'] = classification.quality_score
            result_df.at[i, 'risk_score'] = classification.risk_score
            result_df.at[i, 'confidence'] = classification.confidence
            result_df.at[i, 'reason_codes'] = ', '.join(classification.reason_codes)
            result_df.at[i, 'temporal_anomaly'] = classification.temporal_anomaly
            result_df.at[i, 'geographic_anomaly'] = classification.geographic_anomaly
            result_df.at[i, 'device_anomaly'] = classification.device_anomaly
            result_df.at[i, 'behavioral_anomaly'] = classification.behavioral_anomaly
            result_df.at[i, 'volume_anomaly'] = classification.volume_anomaly
            result_df.at[i, 'overall_anomaly_count'] = classification.overall_anomaly_count
        
        # Log classification summary
        fraud_count = len(result_df[result_df['classification'] == 'fraud'])
        good_count = len(result_df[result_df['classification'] == 'good_account'])
        avg_quality = result_df['quality_score'].mean()
        avg_risk = result_df['risk_score'].mean()
        
        logger.info(f"Classification Summary:")
        logger.info(f"  Total rows: {len(result_df)}")
        logger.info(f"  Fraud: {fraud_count} ({fraud_count/len(result_df)*100:.1f}%)")
        logger.info(f"  Good accounts: {good_count} ({good_count/len(result_df)*100:.1f}%)")
        logger.info(f"  Average quality score: {avg_quality:.2f}")
        logger.info(f"  Average risk score: {avg_risk:.3f}")
        
        return result_df
    
    def _create_quality_mapping(self, quality_results: pd.DataFrame) -> Dict:
        """Create mapping from channelId to quality metrics"""
        if quality_results.empty:
            logger.warning("Empty quality results, using default scores")
            return {}
        
        # Handle different quality result formats
        if 'channelId' in quality_results.columns:
            quality_mapping = quality_results.set_index('channelId').to_dict('index')
        elif quality_results.index.name == 'channelId':
            quality_mapping = quality_results.to_dict('index')
        else:
            # Use index as channelId
            quality_mapping = quality_results.to_dict('index')
        
        logger.info(f"Created quality mapping for {len(quality_mapping)} channels")
        return quality_mapping
    
    def _create_anomaly_mapping(self, anomaly_results: pd.DataFrame, original_df: pd.DataFrame) -> Dict:
        """Create anomaly mapping - handle both row-level and channel-level results"""
        if anomaly_results.empty:
            logger.warning("Empty anomaly results, using default values")
            return {}
        
        anomaly_mapping = {}
        
        # Check if anomaly results are row-level (same length as original data)
        if len(anomaly_results) == len(original_df):
            logger.info("Using row-level anomaly results")
            anomaly_mapping = anomaly_results.to_dict('index')
        elif 'channelId' in anomaly_results.columns:
            logger.info("Using channel-level anomaly results")
            anomaly_mapping = anomaly_results.set_index('channelId').to_dict('index')
        elif anomaly_results.index.name == 'channelId':
            logger.info("Using channel-level anomaly results from index")
            anomaly_mapping = anomaly_results.to_dict('index')
        else:
            logger.warning("Cannot determine anomaly result structure, using as row-level")
            anomaly_mapping = anomaly_results.to_dict('index')
        
        logger.info(f"Created anomaly mapping for {len(anomaly_mapping)} entries")
        return anomaly_mapping
    
    def _classify_batch(self, 
                       batch: pd.DataFrame,
                       quality_mapping: Dict,
                       anomaly_mapping: Dict,
                       features_df: pd.DataFrame = None) -> List[ClassificationResult]:
        """Classify a batch of rows"""
        results = []
        
        for idx, row in batch.iterrows():
            try:
                classification = self._classify_single_row(
                    row, idx, quality_mapping, anomaly_mapping, features_df
                )
                results.append(classification)
            except Exception as e:
                logger.warning(f"Error classifying row {idx}: {e}")
                # Return default classification
                results.append(ClassificationResult(
                    classification='good_account',
                    quality_score=5.0,
                    risk_score=0.0,
                    confidence=0.1,
                    reason_codes=[FraudReason.INSUFFICIENT_DATA.value],
                    temporal_anomaly=False,
                    geographic_anomaly=False,
                    device_anomaly=False,
                    behavioral_anomaly=False,
                    volume_anomaly=False,
                    overall_anomaly_count=0
                ))
        
        return results
    
    def _classify_single_row(self, 
                            row: pd.Series,
                            row_idx: int,
                            quality_mapping: Dict,
                            anomaly_mapping: Dict,
                            features_df: pd.DataFrame = None) -> ClassificationResult:
        """Classify a single row using business rules and ensemble approach"""
        
        # Extract channel ID
        channel_id = row.get('channelId', '')
        
        # Get quality score for this channel
        channel_quality = quality_mapping.get(channel_id, {})
        quality_score = channel_quality.get('quality_score', 5.0)
        
        # Get anomaly indicators (try both row-level and channel-level)
        row_anomalies = anomaly_mapping.get(row_idx, {})
        channel_anomalies = anomaly_mapping.get(channel_id, {})
        
        # Prefer row-level anomalies, fall back to channel-level
        anomalies = row_anomalies if row_anomalies else channel_anomalies
        
        # Extract anomaly flags
        temporal_anomaly = anomalies.get('temporal_anomaly', False)
        geographic_anomaly = anomalies.get('geographic_anomaly', False)
        device_anomaly = anomalies.get('device_anomaly', False)
        behavioral_anomaly = anomalies.get('behavioral_anomaly', False)
        volume_anomaly = anomalies.get('volume_anomaly', False)
        overall_anomaly_count = anomalies.get('overall_anomaly_count', 0)
        
        # Extract individual indicators from row data
        is_bot = row.get('isLikelyBot', False) or row.get('isBot', False)
        is_datacenter = row.get('isIpDatacenter', False)
        is_anonymous = row.get('isIpAnonymous', False)
        ip_classification = row.get('ipClassification', '')
        
        # Calculate risk score using ensemble approach
        risk_score = self._calculate_risk_score(
            quality_score, overall_anomaly_count, is_bot, is_datacenter, is_anonymous, ip_classification
        )
        
        # Determine classification using business rules
        classification, reason_codes, confidence = self._apply_business_rules(
            quality_score, overall_anomaly_count, risk_score, is_bot, is_datacenter, 
            is_anonymous, ip_classification, temporal_anomaly, geographic_anomaly,
            device_anomaly, behavioral_anomaly, volume_anomaly
        )
        
        return ClassificationResult(
            classification=classification,
            quality_score=quality_score,
            risk_score=risk_score,
            confidence=confidence,
            reason_codes=reason_codes,
            temporal_anomaly=temporal_anomaly,
            geographic_anomaly=geographic_anomaly,
            device_anomaly=device_anomaly,
            behavioral_anomaly=behavioral_anomaly,
            volume_anomaly=volume_anomaly,
            overall_anomaly_count=overall_anomaly_count
        )
    
    def _calculate_risk_score(self, 
                             quality_score: float,
                             anomaly_count: int,
                             is_bot: bool,
                             is_datacenter: bool,
                             is_anonymous: bool,
                             ip_classification: str) -> float:
        """Calculate risk score using weighted ensemble"""
        
        # Normalize quality score (invert so higher = more risk)
        quality_risk = (10 - quality_score) / 10.0
        
        # Normalize anomaly count (cap at 5 for scoring)
        anomaly_risk = min(anomaly_count / 5.0, 1.0)
        
        # Bot indicator risk
        bot_risk = 1.0 if is_bot else 0.0
        
        # IP-based risk
        ip_risk = 0.0
        if is_datacenter:
            ip_risk += 0.5
        if is_anonymous:
            ip_risk += 0.3
        if ip_classification in ['suspicious', 'malicious']:
            ip_risk += 0.4
        ip_risk = min(ip_risk, 1.0)
        
        # Weighted ensemble
        risk_score = (
            self.weights['quality_score'] * quality_risk +
            self.weights['anomaly_count'] * anomaly_risk +
            self.weights['bot_indicator'] * bot_risk +
            self.weights['ip_risk'] * ip_risk
        )
        
        return min(max(risk_score, 0.0), 1.0)
    
    def _apply_business_rules(self, 
                             quality_score: float,
                             anomaly_count: int,
                             risk_score: float,
                             is_bot: bool,
                             is_datacenter: bool,
                             is_anonymous: bool,
                             ip_classification: str,
                             temporal_anomaly: bool,
                             geographic_anomaly: bool,
                             device_anomaly: bool,
                             behavioral_anomaly: bool,
                             volume_anomaly: bool) -> Tuple[str, List[str], float]:
        """Apply business rules for classification"""
        
        reason_codes = []
        classification = "good_account"
        confidence = 0.5
        
        # Rule 1: High quality + low anomalies = good account
        if quality_score >= self.quality_threshold_high and anomaly_count <= 1:
            classification = "good_account"
            reason_codes.append(FraudReason.CLEAN_PATTERN.value)
            confidence = 0.8
        
        # Rule 2: Low quality + high anomalies = fraud
        elif quality_score <= self.quality_threshold_low and anomaly_count >= self.anomaly_threshold_high:
            classification = "fraud"
            reason_codes.append(FraudReason.MULTIPLE_INDICATORS.value)
            confidence = 0.9
        
        # Rule 3: Bot activity
        elif is_bot:
            classification = "fraud"
            reason_codes.append(FraudReason.HIGH_BOT_ACTIVITY.value)
            confidence = 0.8
        
        # Rule 4: Datacenter IP
        elif is_datacenter:
            classification = "fraud"
            reason_codes.append(FraudReason.DATACENTER_IP.value)
            confidence = 0.7
        
        # Rule 5: Anonymous IP
        elif is_anonymous:
            classification = "fraud"
            reason_codes.append(FraudReason.ANONYMOUS_IP.value)
            confidence = 0.6
        
        # Rule 6: High risk score
        elif risk_score >= self.risk_threshold:
            classification = "fraud"
            confidence = min(0.5 + risk_score * 0.4, 0.9)
        
        # Rule 7: Low quality score alone
        elif quality_score <= self.quality_threshold_low:
            classification = "fraud"
            reason_codes.append(FraudReason.LOW_QUALITY_SCORE.value)
            confidence = 0.6
        
        # Rule 8: Multiple anomalies
        elif anomaly_count >= self.anomaly_threshold_high:
            classification = "fraud"
            reason_codes.append(FraudReason.MULTIPLE_INDICATORS.value)
            confidence = 0.7
        
        # Add specific anomaly reasons
        if temporal_anomaly:
            reason_codes.append(FraudReason.TEMPORAL_ANOMALY.value)
        if geographic_anomaly:
            reason_codes.append(FraudReason.GEOGRAPHIC_ANOMALY.value)
        if device_anomaly:
            reason_codes.append(FraudReason.DEVICE_ANOMALY.value)
        if behavioral_anomaly:
            reason_codes.append(FraudReason.BEHAVIORAL_ANOMALY.value)
        if volume_anomaly:
            reason_codes.append(FraudReason.VOLUME_ANOMALY.value)
        
        # Add IP pattern reasons
        if ip_classification in ['suspicious', 'malicious']:
            reason_codes.append(FraudReason.SUSPICIOUS_IP_PATTERN.value)
        
        # Default to clean pattern if no reasons identified
        if not reason_codes and classification == "good_account":
            reason_codes.append(FraudReason.CLEAN_PATTERN.value)
        
        return classification, reason_codes, confidence

    def get_classification_thresholds(self) -> Dict[str, float]:
        """Return current classification thresholds for documentation"""
        return {
            'quality_threshold_low': self.quality_threshold_low,
            'quality_threshold_high': self.quality_threshold_high,
            'anomaly_threshold_high': self.anomaly_threshold_high,
            'risk_threshold': self.risk_threshold,
            'weights': self.weights
        }