"""
Traffic Similarity Model
Unsupervised learning models to identify channels with similar traffic patterns.
Uses clustering and embeddings to group channels by behavior.
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.neighbors import NearestNeighbors
import umap
import logging
from typing import Dict, List, Tuple, Optional
import pickle
import joblib

logger = logging.getLogger(__name__)

class TrafficSimilarityModel:
    """
    Unsupervised model to identify channels with similar traffic patterns.
    Uses multiple clustering approaches and dimensionality reduction techniques.
    """
    
    def __init__(self, n_clusters: int = 10, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.scaler = None
        self.models = {}
        self.embeddings = {}
        self.feature_importance = {}
        
    def prepare_features(self, channel_features: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare channel features for similarity analysis with robust error handling.
        
        Args:
            channel_features: DataFrame with channel-level aggregated features
            
        Returns:
            Processed feature matrix
        """
        logger.info("Preparing features for similarity analysis")
        
        if channel_features.empty:
            logger.warning("Empty channel_features DataFrame provided")
            return pd.DataFrame()
        
        # Remove non-numeric columns and handle missing values
        numeric_features = channel_features.select_dtypes(include=[np.number])
        
        if numeric_features.empty:
            logger.warning("No numeric features found in channel_features")
            return pd.DataFrame()
        
        numeric_features = numeric_features.fillna(numeric_features.median())
        
        # Remove features with low variance (with safety check)
        variance_threshold = 0.01
        feature_variances = numeric_features.var()
        high_variance_features = feature_variances[feature_variances > variance_threshold].index
        
        if len(high_variance_features) == 0:
            logger.warning(f"All features have variance <= {variance_threshold}. Using all features with fallback variance threshold.")
            # Fallback: use all features if none meet the variance threshold
            high_variance_features = numeric_features.columns
        
        numeric_features = numeric_features[high_variance_features]
        
        # Remove highly correlated features (with safety check)
        if len(numeric_features.columns) > 1:
            correlation_matrix = numeric_features.corr().abs()
            upper_triangle = correlation_matrix.where(
                np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
            )
            
            high_corr_features = [
                column for column in upper_triangle.columns 
                if any(upper_triangle[column] > 0.95)
            ]
            
            # Keep at least one feature even if highly correlated
            if len(high_corr_features) < len(numeric_features.columns):
                numeric_features = numeric_features.drop(columns=high_corr_features)
            else:
                logger.warning("All features are highly correlated. Keeping first feature as fallback.")
                numeric_features = numeric_features.iloc[:, [0]]
        
        # Final safety check
        if numeric_features.empty or len(numeric_features.columns) == 0:
            logger.error("No valid features remain after preprocessing")
            return pd.DataFrame()
        
        logger.info(f"Feature preparation complete. Using {numeric_features.shape[1]} features out of {channel_features.shape[1]} original columns")
        return numeric_features
    
    def fit(self, channel_features: pd.DataFrame) -> Dict:
        """
        Fit multiple clustering models to identify similar traffic patterns.
        
        Args:
            channel_features: Channel-level features
            
        Returns:
            Dictionary with model results and metrics
        """
        logger.info("Fitting traffic similarity models")
        
        # Early validation of input
        if channel_features.empty:
            logger.warning("Empty channel_features provided to traffic similarity model")
            return self._create_empty_results()
        
        # Prepare features with robust error handling
        try:
            X = self.prepare_features(channel_features)
        except Exception as e:
            logger.error(f"Feature preparation failed: {e}")
            return self._create_empty_results()
        
        # Handle empty features case with more comprehensive checks
        if X.empty or len(X.columns) == 0 or len(X) == 0:
            logger.warning("No valid features available for traffic similarity analysis")
            return self._create_empty_results()
        
        # Check if we have enough data points for clustering
        if len(X) < 2:
            logger.warning(f"Insufficient data for clustering: only {len(X)} samples")
            return self._create_empty_results()
        
        # Scale features
        self.scaler = RobustScaler()
        try:
            X_scaled = self.scaler.fit_transform(X)
        except Exception as e:
            logger.error(f"Failed to scale features: {e}")
            return {
                'error': f'Feature scaling failed: {e}',
                'kmeans': {'labels': [], 'silhouette_score': -1},
                'dbscan': {'labels': [], 'n_clusters': 0, 'n_outliers': 0, 'silhouette_score': -1},
                'hierarchical': {'labels': [], 'silhouette_score': -1}
            }
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        results = {}
        
        # 1. K-Means Clustering
        logger.info("Fitting K-Means clustering")
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init=10)
        kmeans_labels = kmeans.fit_predict(X_scaled)
        
        self.models['kmeans'] = kmeans
        results['kmeans'] = {
            'labels': kmeans_labels,
            'silhouette_score': silhouette_score(X_scaled, kmeans_labels),
            'calinski_harabasz_score': calinski_harabasz_score(X_scaled, kmeans_labels),
            'inertia': kmeans.inertia_
        }
        
        # 2. DBSCAN Clustering (density-based)
        logger.info("Fitting DBSCAN clustering")
        # Find optimal eps using k-distance graph
        eps = self._find_optimal_eps(X_scaled)
        dbscan = DBSCAN(eps=eps, min_samples=5)
        dbscan_labels = dbscan.fit_predict(X_scaled)
        
        self.models['dbscan'] = dbscan
        if len(set(dbscan_labels)) > 1:  # Check if clustering found multiple clusters
            results['dbscan'] = {
                'labels': dbscan_labels,
                'n_clusters': len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0),
                'n_outliers': sum(dbscan_labels == -1),
                'silhouette_score': silhouette_score(X_scaled, dbscan_labels) if len(set(dbscan_labels)) > 1 else -1
            }
        
        # 3. Hierarchical Clustering
        logger.info("Fitting Hierarchical clustering")
        hierarchical = AgglomerativeClustering(n_clusters=self.n_clusters, linkage='ward')
        hierarchical_labels = hierarchical.fit_predict(X_scaled)
        
        self.models['hierarchical'] = hierarchical
        results['hierarchical'] = {
            'labels': hierarchical_labels,
            'silhouette_score': silhouette_score(X_scaled, hierarchical_labels),
            'calinski_harabasz_score': calinski_harabasz_score(X_scaled, hierarchical_labels)
        }
        
        # 4. Create embeddings for visualization and similarity search
        logger.info("Creating embeddings")
        self.embeddings = self._create_embeddings(X_scaled)
        
        # 5. Feature importance analysis
        self.feature_importance = self._analyze_feature_importance(X, X_scaled, kmeans_labels)
        
        logger.info("Traffic similarity model fitting complete")
        return results
    
    def _find_optimal_eps(self, X: np.ndarray, k: int = 5) -> float:
        """Find optimal eps parameter for DBSCAN using k-distance graph."""
        neighbors = NearestNeighbors(n_neighbors=k)
        neighbors_fit = neighbors.fit(X)
        distances, indices = neighbors_fit.kneighbors(X)
        
        # Sort distances to kth nearest neighbor
        distances = np.sort(distances[:, k-1], axis=0)
        
        # Use elbow method - simplified version
        # In production, you might want a more sophisticated elbow detection
        eps = np.percentile(distances, 95)  # Use 95th percentile as eps
        return eps
    
    def _create_embeddings(self, X: np.ndarray) -> Dict:
        """Create various embeddings for visualization and similarity search."""
        embeddings = {}
        
        # PCA
        pca = PCA(n_components=min(10, X.shape[1]), random_state=self.random_state)
        embeddings['pca'] = pca.fit_transform(X)
        self.models['pca'] = pca
        
        # UMAP
        try:
            umap_model = umap.UMAP(n_components=2, random_state=self.random_state, n_neighbors=15)
            embeddings['umap'] = umap_model.fit_transform(X)
            self.models['umap'] = umap_model
        except Exception as e:
            logger.warning(f"UMAP embedding failed: {e}")
        
        # t-SNE (for visualization)
        if X.shape[0] < 5000:  # t-SNE is computationally expensive
            try:
                tsne = TSNE(n_components=2, random_state=self.random_state, perplexity=min(30, X.shape[0]//4))
                embeddings['tsne'] = tsne.fit_transform(X)
            except Exception as e:
                logger.warning(f"t-SNE embedding failed: {e}")
        
        return embeddings
    
    def _analyze_feature_importance(self, X: pd.DataFrame, X_scaled: np.ndarray, labels: np.ndarray) -> Dict:
        """Analyze which features are most important for clustering."""
        feature_importance = {}
        
        # Calculate feature importance based on cluster separation
        from sklearn.metrics import f1_score
        from sklearn.ensemble import RandomForestClassifier
        
        # Use random forest to identify important features for cluster prediction
        rf = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
        rf.fit(X_scaled, labels)
        
        feature_importance['random_forest'] = dict(zip(self.feature_names, rf.feature_importances_))
        
        # Calculate variance between clusters for each feature
        cluster_means = {}
        for cluster in np.unique(labels):
            cluster_mask = labels == cluster
            cluster_means[cluster] = X_scaled[cluster_mask].mean(axis=0)
        
        # Calculate inter-cluster variance for each feature
        inter_cluster_variance = np.var([means for means in cluster_means.values()], axis=0)
        feature_importance['inter_cluster_variance'] = dict(zip(self.feature_names, inter_cluster_variance))
        
        return feature_importance
    
    def _create_empty_results(self) -> Dict:
        """Create standardized empty results for error cases"""
        return {
            'error': 'No valid features or insufficient data',
            'similar_pairs': [],
            'num_channels': 0,
            'similarity_threshold': 0.5,
            'kmeans': {'labels': [], 'silhouette_score': -1},
            'dbscan': {'labels': [], 'n_clusters': 0, 'n_outliers': 0, 'silhouette_score': -1},
            'hierarchical': {'labels': [], 'silhouette_score': -1}
        }
    
    def find_similar_channels(self, channel_id: str, channel_features: pd.DataFrame, 
                            method: str = 'kmeans', n_similar: int = 5) -> List[Tuple[str, float]]:
        """
        Find channels most similar to a given channel.
        
        Args:
            channel_id: Target channel ID
            channel_features: Channel features DataFrame
            method: Clustering method to use ('kmeans', 'dbscan', 'hierarchical')
            n_similar: Number of similar channels to return
            
        Returns:
            List of (channel_id, similarity_score) tuples
        """
        if method not in self.models:
            raise ValueError(f"Method {method} not available. Available methods: {list(self.models.keys())}")
        
        if channel_id not in channel_features.index:
            raise ValueError(f"Channel {channel_id} not found in features")
        
        # Prepare features
        X = self.prepare_features(channel_features)
        X_scaled = self.scaler.transform(X)
        
        # Get target channel features
        target_idx = X.index.get_loc(channel_id)
        target_features = X_scaled[target_idx].reshape(1, -1)
        
        # Find similar channels based on clustering
        if method == 'kmeans':
            # Get cluster assignment
            target_cluster = self.models['kmeans'].predict(target_features)[0]
            cluster_labels = self.models['kmeans'].labels_
            
            # Find all channels in the same cluster
            same_cluster_mask = cluster_labels == target_cluster
            same_cluster_indices = np.where(same_cluster_mask)[0]
            
            # Calculate distances to channels in the same cluster
            distances = []
            for idx in same_cluster_indices:
                if idx != target_idx:
                    distance = np.linalg.norm(X_scaled[idx] - X_scaled[target_idx])
                    distances.append((X.index[idx], distance))
            
            # Sort by distance and return top n
            distances.sort(key=lambda x: x[1])
            return distances[:n_similar]
        
        elif method in ['dbscan', 'hierarchical']:
            # For other methods, use k-nearest neighbors in feature space
            from sklearn.neighbors import NearestNeighbors
            nn = NearestNeighbors(n_neighbors=n_similar + 1, metric='euclidean')
            nn.fit(X_scaled)
            
            distances, indices = nn.kneighbors(target_features)
            
            # Skip the first result (the channel itself)
            similar_channels = []
            for i in range(1, len(indices[0])):
                idx = indices[0][i]
                distance = distances[0][i]
                similar_channels.append((X.index[idx], 1.0 / (1.0 + distance)))  # Convert to similarity score
            
            return similar_channels
    
    def get_cluster_profiles(self, channel_features: pd.DataFrame, method: str = 'kmeans') -> Dict:
        """
        Get profiles of each cluster to understand traffic patterns.
        
        Args:
            channel_features: Channel features DataFrame
            method: Clustering method to use
            
        Returns:
            Dictionary with cluster profiles
        """
        if method not in self.models:
            raise ValueError(f"Method {method} not available")
        
        X = self.prepare_features(channel_features)
        
        if method == 'kmeans':
            labels = self.models['kmeans'].labels_
        elif method == 'hierarchical':
            # Re-predict labels for hierarchical clustering
            X_scaled = self.scaler.transform(X)
            labels = self.models['hierarchical'].fit_predict(X_scaled)
        else:
            raise ValueError(f"Cluster profiling not implemented for {method}")
        
        cluster_profiles = {}
        
        for cluster in np.unique(labels):
            cluster_mask = labels == cluster
            cluster_data = X[cluster_mask]
            
            profile = {
                'size': cluster_mask.sum(),
                'channels': X.index[cluster_mask].tolist(),
                'mean_features': cluster_data.mean().to_dict(),
                'std_features': cluster_data.std().to_dict()
            }
            
            # Identify top distinguishing features
            overall_mean = X.mean()
            feature_deviations = abs(cluster_data.mean() - overall_mean) / X.std()
            top_features = feature_deviations.nlargest(5).index.tolist()
            profile['distinguishing_features'] = top_features
            
            cluster_profiles[f'cluster_{cluster}'] = profile
        
        return cluster_profiles
    
    def detect_outlier_channels(self, channel_features: pd.DataFrame, contamination: float = 0.1) -> List[str]:
        """
        Detect channels with unusual traffic patterns using Isolation Forest.
        
        Args:
            channel_features: Channel features DataFrame
            contamination: Expected proportion of outliers
            
        Returns:
            List of channel IDs identified as outliers
        """
        logger.info("Detecting outlier channels")
        
        X = self.prepare_features(channel_features)
        X_scaled = self.scaler.transform(X)
        
        # Fit Isolation Forest
        isolation_forest = IsolationForest(
            contamination=contamination,
            random_state=self.random_state,
            n_estimators=100
        )
        outlier_labels = isolation_forest.fit_predict(X_scaled)
        
        # Get outlier channels
        outlier_mask = outlier_labels == -1
        outlier_channels = X.index[outlier_mask].tolist()
        
        self.models['isolation_forest'] = isolation_forest
        
        logger.info(f"Detected {len(outlier_channels)} outlier channels")
        return outlier_channels
    
    def save_model(self, filepath: str):
        """Save the trained model."""
        model_data = {
            'scaler': self.scaler,
            'models': self.models,
            'embeddings': self.embeddings,
            'feature_importance': self.feature_importance,
            'feature_names': self.feature_names,
            'n_clusters': self.n_clusters
        }
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model."""
        model_data = joblib.load(filepath)
        self.scaler = model_data['scaler']
        self.models = model_data['models']
        self.embeddings = model_data['embeddings']
        self.feature_importance = model_data['feature_importance']
        self.feature_names = model_data['feature_names']
        self.n_clusters = model_data['n_clusters']
        logger.info(f"Model loaded from {filepath}")

def main():
    """Test traffic similarity model."""
    from data_pipeline import DataPipeline
    from feature_engineering import FeatureEngineer
    
    # Load sample data
    pipeline = DataPipeline("/home/fiod/shimshi/bq-results-20250804-141411-1754316868932.csv")
    df = pipeline.load_data_chunked(sample_fraction=0.05)  # 5% sample
    
    # Engineer features
    feature_engineer = FeatureEngineer()
    features_df = feature_engineer.create_all_features(df)
    channel_features = feature_engineer.create_channel_features(features_df)
    
    # Fit similarity model
    similarity_model = TrafficSimilarityModel(n_clusters=8)
    results = similarity_model.fit(channel_features)
    
    # Print results
    logger.info("Clustering Results:")
    for method, result in results.items():
        if 'silhouette_score' in result:
            logger.info(f"{method}: Silhouette Score = {result['silhouette_score']:.3f}")
    
    # Get cluster profiles
    profiles = similarity_model.get_cluster_profiles(channel_features)
    logger.info(f"Created {len(profiles)} cluster profiles")
    
    # Detect outliers
    outliers = similarity_model.detect_outlier_channels(channel_features)
    logger.info(f"Detected {len(outliers)} outlier channels")
    
    # Save model
    similarity_model.save_model("/home/fiod/shimshi/traffic_similarity_model.pkl")
    
    return similarity_model, results, profiles, outliers

if __name__ == "__main__":
    similarity_model, results, profiles, outliers = main()