"""
Production Model Serving Infrastructure
Flask-based API for serving fraud detection and quality scoring models in production.
Includes caching, monitoring, and scalable inference capabilities.
"""

from flask import Flask, request, jsonify, Response
import pandas as pd
import numpy as np
import logging
import joblib
import json
from datetime import datetime, timedelta
import redis
import hashlib
from typing import Dict, List, Optional, Any
import os
from dataclasses import dataclass, asdict
import threading
import time
from functools import wraps
import warnings
warnings.filterwarnings('ignore')

# Import our ML models
from data_pipeline import DataPipeline
from feature_engineering import FeatureEngineer
from quality_scoring import QualityScorer
from traffic_similarity import TrafficSimilarityModel
from anomaly_detection import AnomalyDetector
from model_evaluation import ModelEvaluator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class PredictionRequest:
    """Structure for prediction requests."""
    channel_id: str
    traffic_data: List[Dict]
    model_types: List[str] = None  # ['quality', 'similarity', 'anomaly']
    use_cache: bool = True

@dataclass
class PredictionResponse:
    """Structure for prediction responses."""
    channel_id: str
    quality_score: Optional[float] = None
    quality_category: Optional[str] = None
    similar_channels: Optional[List[Tuple[str, float]]] = None
    anomaly_flags: Optional[Dict[str, bool]] = None
    anomaly_score: Optional[float] = None
    risk_level: Optional[str] = None
    timestamp: str = None
    processing_time_ms: float = 0

class ModelServer:
    """
    Production model serving infrastructure with caching, monitoring, and scaling.
    """
    
    def __init__(self, model_dir: str = "/home/fiod/shimshi/", use_redis: bool = False):
        self.model_dir = model_dir
        self.use_redis = use_redis
        self.models = {}
        self.feature_engineer = None
        self.cache = {}
        self.redis_client = None
        self.request_count = 0
        self.request_times = []
        self.model_load_time = None
        
        # Initialize Redis if available
        if use_redis:
            try:
                self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
                self.redis_client.ping()
                logger.info("Redis cache initialized")
            except:
                logger.warning("Redis not available, using in-memory cache")
                self.redis_client = None
        
        self.load_models()
        
    def load_models(self):
        """Load all trained models."""
        start_time = time.time()
        logger.info("Loading trained models...")
        
        try:
            # Load feature engineer
            self.feature_engineer = FeatureEngineer()
            
            # Load quality scoring model
            quality_scorer = QualityScorer()
            quality_model_path = os.path.join(self.model_dir, "quality_scoring_model.pkl")
            if os.path.exists(quality_model_path):
                quality_scorer.load_model(quality_model_path)
                self.models['quality'] = quality_scorer
                logger.info("Quality scoring model loaded")
            
            # Load similarity model
            similarity_model_path = os.path.join(self.model_dir, "traffic_similarity_model.pkl")
            if os.path.exists(similarity_model_path):
                similarity_model = TrafficSimilarityModel()
                similarity_model.load_model(similarity_model_path)
                self.models['similarity'] = similarity_model
                logger.info("Similarity model loaded")
            
            # Load anomaly detection model
            anomaly_model_path = os.path.join(self.model_dir, "anomaly_detection_model.pkl")
            if os.path.exists(anomaly_model_path):
                anomaly_detector = AnomalyDetector()
                anomaly_detector.load_model(anomaly_model_path)
                self.models['anomaly'] = anomaly_detector
                logger.info("Anomaly detection model loaded")
            
            self.model_load_time = time.time() - start_time
            logger.info(f"All models loaded in {self.model_load_time:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise
    
    def _get_cache_key(self, request_data: Dict) -> str:
        """Generate cache key for request."""
        # Create hash of relevant request data
        cache_data = {
            'channel_id': request_data.get('channel_id'),
            'data_hash': hashlib.md5(str(request_data.get('traffic_data', [])).encode()).hexdigest()
        }
        return hashlib.md5(json.dumps(cache_data, sort_keys=True).encode()).hexdigest()
    
    def _get_from_cache(self, cache_key: str) -> Optional[Dict]:
        """Get prediction from cache."""
        try:
            if self.redis_client:
                cached = self.redis_client.get(cache_key)
                if cached:
                    return json.loads(cached)
            else:
                return self.cache.get(cache_key)
        except Exception as e:
            logger.warning(f"Cache retrieval error: {e}")
        return None
    
    def _set_cache(self, cache_key: str, prediction: Dict, ttl: int = 3600):
        """Set prediction in cache."""
        try:
            if self.redis_client:
                self.redis_client.setex(cache_key, ttl, json.dumps(prediction, default=str))
            else:
                self.cache[cache_key] = prediction
                # Simple in-memory cache cleanup (keep last 1000 entries)
                if len(self.cache) > 1000:
                    keys_to_remove = list(self.cache.keys())[:-900]
                    for key in keys_to_remove:
                        del self.cache[key]
        except Exception as e:
            logger.warning(f"Cache storage error: {e}")
    
    def preprocess_traffic_data(self, traffic_data: List[Dict]) -> pd.DataFrame:
        """
        Convert raw traffic data to DataFrame and engineer features.
        
        Args:
            traffic_data: List of traffic records
            
        Returns:
            DataFrame with engineered features
        """
        # Convert to DataFrame
        df = pd.DataFrame(traffic_data)
        
        # Basic data validation and cleaning
        required_columns = ['date', 'channelId']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Convert date column
        df['date'] = pd.to_datetime(df['date'])
        
        # Engineer features
        features_df = self.feature_engineer.create_all_features(df)
        
        return features_df
    
    def predict_quality_score(self, features_df: pd.DataFrame, channel_id: str) -> Dict:
        """Predict quality score for a channel."""
        if 'quality' not in self.models:
            return {'error': 'Quality scoring model not available'}
        
        try:
            quality_scorer = self.models['quality']
            
            # Create channel features if not exists
            if not hasattr(quality_scorer, 'channel_features') or quality_scorer.channel_features.empty:
                channel_features = self.feature_engineer.create_channel_features(features_df)
                quality_scorer.channel_features = channel_features
            
            # Score the specific channel
            results_df = quality_scorer.score_channels(features_df)
            
            if channel_id in results_df.index:
                channel_result = results_df.loc[channel_id]
                return {
                    'quality_score': float(channel_result['quality_score']),
                    'quality_category': str(channel_result['quality_category']),
                    'high_risk': bool(channel_result['high_risk']),
                    'bot_rate': float(channel_result['bot_rate']),
                    'fraud_score': float(channel_result['fraud_score_avg'])
                }
            else:
                return {'error': f'Channel {channel_id} not found in results'}
                
        except Exception as e:
            logger.error(f"Quality prediction error: {e}")
            return {'error': str(e)}
    
    def find_similar_channels(self, features_df: pd.DataFrame, channel_id: str, n_similar: int = 5) -> Dict:
        """Find similar channels."""
        if 'similarity' not in self.models:
            return {'error': 'Similarity model not available'}
        
        try:
            similarity_model = self.models['similarity']
            channel_features = self.feature_engineer.create_channel_features(features_df)
            
            if channel_id not in channel_features.index:
                return {'error': f'Channel {channel_id} not found'}
            
            similar_channels = similarity_model.find_similar_channels(
                channel_id, channel_features, n_similar=n_similar
            )
            
            return {
                'similar_channels': [(ch_id, float(score)) for ch_id, score in similar_channels]
            }
            
        except Exception as e:
            logger.error(f"Similarity prediction error: {e}")
            return {'error': str(e)}
    
    def detect_anomalies(self, features_df: pd.DataFrame, channel_id: str) -> Dict:
        """Detect anomalies for a channel."""
        if 'anomaly' not in self.models:
            return {'error': 'Anomaly detection model not available'}
        
        try:
            anomaly_detector = self.models['anomaly']
            anomaly_results = anomaly_detector.run_comprehensive_anomaly_detection(features_df)
            
            if not anomaly_results.empty and channel_id in anomaly_results.index:
                channel_anomalies = anomaly_results.loc[channel_id]
                
                # Extract anomaly flags
                anomaly_flags = {}
                for col in anomaly_results.columns:
                    if 'anomaly' in col and anomaly_results[col].dtype == bool:
                        anomaly_flags[col] = bool(channel_anomalies[col])
                
                overall_anomaly_score = 0
                if 'overall_anomaly_count' in channel_anomalies:
                    overall_anomaly_score = float(channel_anomalies['overall_anomaly_count'])
                
                return {
                    'anomaly_flags': anomaly_flags,
                    'anomaly_score': overall_anomaly_score,
                    'is_anomalous': overall_anomaly_score >= 2
                }
            else:
                return {'error': f'No anomaly results for channel {channel_id}'}
                
        except Exception as e:
            logger.error(f"Anomaly detection error: {e}")
            return {'error': str(e)}
    
    def predict(self, request_data: Dict) -> Dict:
        """
        Main prediction method that orchestrates all model predictions.
        
        Args:
            request_data: Dictionary with prediction request data
            
        Returns:
            Dictionary with prediction results
        """
        start_time = time.time()
        
        try:
            # Parse request
            channel_id = request_data['channel_id']
            traffic_data = request_data['traffic_data']
            model_types = request_data.get('model_types', ['quality', 'similarity', 'anomaly'])
            use_cache = request_data.get('use_cache', True)
            
            # Check cache
            cache_key = self._get_cache_key(request_data)
            if use_cache:
                cached_result = self._get_from_cache(cache_key)
                if cached_result:
                    cached_result['from_cache'] = True
                    return cached_result
            
            # Preprocess data
            features_df = self.preprocess_traffic_data(traffic_data)
            
            # Initialize response
            response = {
                'channel_id': channel_id,
                'timestamp': datetime.now().isoformat(),
                'from_cache': False
            }
            
            # Run predictions based on requested model types
            if 'quality' in model_types:
                quality_result = self.predict_quality_score(features_df, channel_id)
                response.update(quality_result)
            
            if 'similarity' in model_types:
                similarity_result = self.find_similar_channels(features_df, channel_id)
                response.update(similarity_result)
            
            if 'anomaly' in model_types:
                anomaly_result = self.detect_anomalies(features_df, channel_id)
                response.update(anomaly_result)
            
            # Calculate risk level
            response['risk_level'] = self._calculate_risk_level(response)
            
            # Add processing time
            processing_time = (time.time() - start_time) * 1000
            response['processing_time_ms'] = processing_time
            
            # Cache result
            if use_cache and 'error' not in response:
                self._set_cache(cache_key, response)
            
            # Update metrics
            self.request_count += 1
            self.request_times.append(processing_time)
            if len(self.request_times) > 1000:  # Keep last 1000 request times
                self.request_times = self.request_times[-900:]
            
            return response
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {
                'channel_id': request_data.get('channel_id', 'unknown'),
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
                'processing_time_ms': (time.time() - start_time) * 1000
            }
    
    def _calculate_risk_level(self, response: Dict) -> str:
        """Calculate overall risk level based on all predictions."""
        risk_score = 0
        
        # Quality score contribution
        if 'quality_score' in response:
            quality_score = response['quality_score']
            if quality_score <= 3:
                risk_score += 3
            elif quality_score <= 5:
                risk_score += 2
            elif quality_score <= 7:
                risk_score += 1
        
        # High risk flag contribution
        if response.get('high_risk', False):
            risk_score += 2
        
        # Anomaly contribution
        if 'anomaly_score' in response:
            anomaly_score = response['anomaly_score']
            if anomaly_score >= 3:
                risk_score += 3
            elif anomaly_score >= 2:
                risk_score += 2
            elif anomaly_score >= 1:
                risk_score += 1
        
        # Convert to risk level
        if risk_score >= 5:
            return 'HIGH'
        elif risk_score >= 3:
            return 'MEDIUM'
        elif risk_score >= 1:
            return 'LOW'
        else:
            return 'MINIMAL'
    
    def get_health_status(self) -> Dict:
        """Get server health status."""
        return {
            'status': 'healthy',
            'models_loaded': list(self.models.keys()),
            'total_requests': self.request_count,
            'avg_response_time_ms': np.mean(self.request_times) if self.request_times else 0,
            'cache_type': 'redis' if self.redis_client else 'memory',
            'model_load_time_seconds': self.model_load_time,
            'uptime_seconds': time.time() - (self.model_load_time or 0)
        }
    
    def get_metrics(self) -> Dict:
        """Get detailed metrics."""
        return {
            'requests': {
                'total': self.request_count,
                'recent_response_times': self.request_times[-100:] if self.request_times else [],
                'avg_response_time_ms': np.mean(self.request_times) if self.request_times else 0,
                'p95_response_time_ms': np.percentile(self.request_times, 95) if self.request_times else 0,
                'p99_response_time_ms': np.percentile(self.request_times, 99) if self.request_times else 0
            },
            'models': {
                'loaded': list(self.models.keys()),
                'load_time_seconds': self.model_load_time
            },
            'cache': {
                'type': 'redis' if self.redis_client else 'memory',
                'size': len(self.cache) if not self.redis_client else 'unknown'
            }
        }

# Flask application
app = Flask(__name__)
model_server = None

def init_model_server():
    """Initialize the model server."""
    global model_server
    if model_server is None:
        model_server = ModelServer()

@app.before_first_request
def startup():
    """Initialize model server on first request."""
    init_model_server()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    try:
        if model_server is None:
            init_model_server()
        health_status = model_server.get_health_status()
        return jsonify(health_status), 200
    except Exception as e:
        return jsonify({'status': 'unhealthy', 'error': str(e)}), 500

@app.route('/metrics', methods=['GET'])
def get_metrics():
    """Get server metrics."""
    try:
        if model_server is None:
            return jsonify({'error': 'Model server not initialized'}), 500
        metrics = model_server.get_metrics()
        return jsonify(metrics), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    """Main prediction endpoint."""
    try:
        if model_server is None:
            init_model_server()
        
        request_data = request.get_json()
        if not request_data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        # Validate required fields
        required_fields = ['channel_id', 'traffic_data']
        for field in required_fields:
            if field not in request_data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Make prediction
        result = model_server.predict(request_data)
        
        # Return appropriate status code
        if 'error' in result:
            return jsonify(result), 400
        else:
            return jsonify(result), 200
            
    except Exception as e:
        logger.error(f"Prediction endpoint error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/similar/<channel_id>', methods=['GET'])
def get_similar_channels(channel_id: str):
    """Get similar channels endpoint."""
    try:
        if model_server is None:
            init_model_server()
        
        # This endpoint would need channel features to be pre-computed
        # For now, return a placeholder response
        return jsonify({
            'message': 'Similar channels endpoint requires batch processing',
            'channel_id': channel_id
        }), 501
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    return jsonify({'error': 'Internal server error'}), 500

def create_sample_request() -> Dict:
    """Create a sample request for testing."""
    return {
        'channel_id': 'test-channel-123',
        'traffic_data': [
            {
                'date': '2025-08-04 10:00:00',
                'channelId': 'test-channel-123',
                'publisherId': 'pub-123',
                'advertiserId': 'adv-123',
                'country': 'US',
                'browser': 'chrome',
                'device': 'notMobile',
                'ip': '192.168.1.1',
                'isLikelyBot': False,
                'isIpDatacenter': False,
                'keyword': 'test keyword',
                'referrer': 'https://example.com',
                'userAgent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
        ],
        'model_types': ['quality', 'anomaly'],
        'use_cache': True
    }

if __name__ == '__main__':
    # For development/testing
    logger.info("Starting Fraud Detection Model Server...")
    
    # Test the model server
    server = ModelServer()
    sample_request = create_sample_request()
    
    logger.info("Testing prediction with sample data...")
    result = server.predict(sample_request)
    logger.info(f"Sample prediction result: {result}")
    
    # Start Flask app
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)