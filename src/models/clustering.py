"""
Clustering models for Customer Segmentation
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import joblib
import warnings
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from config import MODELS_DIR, CLUSTERING_MODEL_FILE, RANDOM_STATE, N_CLUSTERS

warnings.filterwarnings('ignore')


class CustomerSegmentation:
    """
    Customer segmentation using clustering algorithms.
    Supports KMeans, DBSCAN, Hierarchical, and Gaussian Mixture Models.
    """
    
    SUPPORTED_MODELS = {
        'kmeans': KMeans,
        'dbscan': DBSCAN,
        'hierarchical': AgglomerativeClustering,
        'gmm': GaussianMixture
    }
    
    DEFAULT_PARAMS = {
        'kmeans': {
            'n_clusters': N_CLUSTERS,
            'random_state': RANDOM_STATE,
            'n_init': 10,
            'max_iter': 300
        },
        'dbscan': {
            'eps': 0.5,
            'min_samples': 5
        },
        'hierarchical': {
            'n_clusters': N_CLUSTERS,
            'linkage': 'ward'
        },
        'gmm': {
            'n_components': N_CLUSTERS,
            'random_state': RANDOM_STATE,
            'covariance_type': 'full'
        }
    }
    
    def __init__(self, model_type: str = 'kmeans', **kwargs):
        """
        Initialize clustering model.
        
        Args:
            model_type: Type of clustering algorithm
            **kwargs: Additional parameters for the model
        """
        if model_type not in self.SUPPORTED_MODELS:
            raise ValueError(f"Model type '{model_type}' not supported. "
                           f"Choose from: {list(self.SUPPORTED_MODELS.keys())}")
        
        self.model_type = model_type
        self.model_class = self.SUPPORTED_MODELS[model_type]
        
        # Merge default params with custom params
        self.params = self.DEFAULT_PARAMS[model_type].copy()
        self.params.update(kwargs)
        
        self.model = self.model_class(**self.params)
        self.is_fitted = False
        self.labels_ = None
        self.cluster_centers_ = None
    
    def fit(self, X: np.ndarray) -> 'CustomerSegmentation':
        """
        Fit the clustering model.
        
        Args:
            X: Features to cluster
        
        Returns:
            self
        """
        print(f"Fitting {self.model_type} clustering model...")
        
        if self.model_type == 'gmm':
            self.model.fit(X)
            self.labels_ = self.model.predict(X)
        else:
            self.labels_ = self.model.fit_predict(X)
        
        # Store cluster centers if available
        if hasattr(self.model, 'cluster_centers_'):
            self.cluster_centers_ = self.model.cluster_centers_
        elif self.model_type == 'gmm':
            self.cluster_centers_ = self.model.means_
        
        self.is_fitted = True
        print(f"Found {len(np.unique(self.labels_))} clusters")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict cluster labels for new data.
        
        Args:
            X: Features to predict
        
        Returns:
            Cluster labels
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted yet!")
        
        if self.model_type == 'kmeans':
            return self.model.predict(X)
        elif self.model_type == 'gmm':
            return self.model.predict(X)
        else:
            # For models that don't support predict, use fit_predict
            print("Warning: This model doesn't support predict on new data.")
            return None
    
    def evaluate(self, X: np.ndarray) -> dict:
        """
        Evaluate clustering quality with multiple metrics.
        
        Args:
            X: Features used for clustering
        
        Returns:
            Dictionary with evaluation metrics
        """
        if self.labels_ is None:
            raise ValueError("Model not fitted yet!")
        
        # Remove noise points for DBSCAN (-1 labels)
        valid_mask = self.labels_ != -1
        if not valid_mask.any():
            print("No valid clusters found!")
            return {}
        
        n_clusters = len(np.unique(self.labels_[valid_mask]))
        
        if n_clusters < 2:
            print("Need at least 2 clusters for evaluation metrics.")
            return {'n_clusters': n_clusters}
        
        X_valid = X[valid_mask]
        labels_valid = self.labels_[valid_mask]
        
        metrics = {
            'n_clusters': n_clusters,
            'silhouette_score': silhouette_score(X_valid, labels_valid),
            'calinski_harabasz_score': calinski_harabasz_score(X_valid, labels_valid),
            'davies_bouldin_score': davies_bouldin_score(X_valid, labels_valid)
        }
        
        # Add inertia for KMeans
        if hasattr(self.model, 'inertia_'):
            metrics['inertia'] = self.model.inertia_
        
        print("\nClustering Evaluation Metrics:")
        print("-" * 40)
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}" if isinstance(value, float) else f"{metric}: {value}")
        
        return metrics
    
    def get_cluster_summary(self, df: pd.DataFrame, 
                            cluster_col: str = 'Cluster') -> pd.DataFrame:
        """
        Generate summary statistics for each cluster.
        
        Args:
            df: DataFrame with features and cluster labels
            cluster_col: Name of cluster column
        
        Returns:
            DataFrame with cluster summaries
        """
        if cluster_col not in df.columns:
            raise ValueError(f"Column '{cluster_col}' not found in DataFrame")
        
        # Numeric columns only
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if cluster_col in numeric_cols:
            numeric_cols.remove(cluster_col)
        
        summary = df.groupby(cluster_col)[numeric_cols].agg(['mean', 'std', 'count'])
        
        return summary
    
    def find_optimal_k(self, X: np.ndarray, k_range: range = range(2, 11)) -> dict:
        """
        Find optimal number of clusters using elbow method and silhouette score.
        
        Args:
            X: Features to cluster
            k_range: Range of k values to try
        
        Returns:
            Dictionary with results for each k
        """
        print("Finding optimal number of clusters...")
        results = {
            'k': [],
            'inertia': [],
            'silhouette': []
        }
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
            labels = kmeans.fit_predict(X)
            
            results['k'].append(k)
            results['inertia'].append(kmeans.inertia_)
            results['silhouette'].append(silhouette_score(X, labels))
            
            print(f"k={k}: inertia={kmeans.inertia_:.2f}, silhouette={results['silhouette'][-1]:.4f}")
        
        # Find best k based on silhouette score
        best_idx = np.argmax(results['silhouette'])
        best_k = results['k'][best_idx]
        print(f"\nOptimal k based on silhouette score: {best_k}")
        
        results['best_k'] = best_k
        return results
    
    def save(self, file_path: Path = None):
        """
        Save the model to disk.
        
        Args:
            file_path: Path to save model
        """
        if file_path is None:
            file_path = MODELS_DIR / CLUSTERING_MODEL_FILE
        
        file_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, file_path)
        print(f"Clustering model saved to {file_path}")
    
    @classmethod
    def load(cls, file_path: Path = None) -> 'CustomerSegmentation':
        """
        Load a model from disk.
        
        Args:
            file_path: Path to load model from
        
        Returns:
            Loaded CustomerSegmentation instance
        """
        if file_path is None:
            file_path = MODELS_DIR / CLUSTERING_MODEL_FILE
        
        model = joblib.load(file_path)
        print(f"Clustering model loaded from {file_path}")
        return model


def create_customer_segments(df: pd.DataFrame, 
                             features: list,
                             n_clusters: int = N_CLUSTERS) -> tuple:
    """
    Create customer segments from data.
    
    Args:
        df: DataFrame with customer data
        features: List of features to use for clustering
        n_clusters: Number of clusters
    
    Returns:
        Tuple of (DataFrame with segments, model, scaler)
    """
    # Prepare data
    X = df[features].values
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Create and fit model
    model = CustomerSegmentation('kmeans', n_clusters=n_clusters)
    model.fit(X_scaled)
    
    # Add segment labels to DataFrame
    df_segmented = df.copy()
    df_segmented['Segment'] = model.labels_
    
    return df_segmented, model, scaler


def analyze_segments(df: pd.DataFrame, segment_col: str = 'Segment') -> pd.DataFrame:
    """
    Analyze characteristics of each customer segment.
    
    Args:
        df: DataFrame with segment labels
        segment_col: Name of segment column
    
    Returns:
        DataFrame with segment analysis
    """
    analysis = df.groupby(segment_col).agg({
        'tenure': 'mean',
        'MonthlyCharges': 'mean',
        'TotalCharges': 'mean',
        'Churn_Binary': 'mean'
    }).round(2)
    
    analysis.columns = ['Avg Tenure', 'Avg Monthly Charges', 
                        'Avg Total Charges', 'Churn Rate']
    
    # Add segment size
    analysis['Size'] = df.groupby(segment_col).size()
    analysis['Size %'] = (analysis['Size'] / len(df) * 100).round(1)
    
    print("\nSegment Analysis:")
    print("=" * 60)
    print(analysis.to_string())
    
    return analysis
