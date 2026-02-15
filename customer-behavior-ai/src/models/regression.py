"""
Regression models (for future expansion - CLV prediction, etc.)
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from config import MODELS_DIR, RANDOM_STATE, CV_FOLDS, N_JOBS


class CLVRegressor:
    """
    Customer Lifetime Value Regressor.
    Can be used to predict total charges or other continuous values.
    """
    
    SUPPORTED_MODELS = {
        'linear': LinearRegression,
        'ridge': Ridge,
        'lasso': Lasso,
        'elastic_net': ElasticNet,
        'random_forest': RandomForestRegressor,
        'gradient_boosting': GradientBoostingRegressor,
        'xgboost': XGBRegressor
    }
    
    DEFAULT_PARAMS = {
        'linear': {},
        'ridge': {
            'alpha': 1.0,
            'random_state': RANDOM_STATE
        },
        'lasso': {
            'alpha': 1.0,
            'random_state': RANDOM_STATE
        },
        'elastic_net': {
            'alpha': 1.0,
            'l1_ratio': 0.5,
            'random_state': RANDOM_STATE
        },
        'random_forest': {
            'random_state': RANDOM_STATE,
            'n_estimators': 100,
            'max_depth': 10,
            'n_jobs': N_JOBS
        },
        'gradient_boosting': {
            'random_state': RANDOM_STATE,
            'n_estimators': 100,
            'max_depth': 5,
            'learning_rate': 0.1
        },
        'xgboost': {
            'random_state': RANDOM_STATE,
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1
        }
    }
    
    def __init__(self, model_type: str = 'xgboost', **kwargs):
        """
        Initialize regressor.
        
        Args:
            model_type: Type of regressor
            **kwargs: Additional parameters
        """
        if model_type not in self.SUPPORTED_MODELS:
            raise ValueError(f"Model type '{model_type}' not supported.")
        
        self.model_type = model_type
        self.model_class = self.SUPPORTED_MODELS[model_type]
        
        self.params = self.DEFAULT_PARAMS[model_type].copy()
        self.params.update(kwargs)
        
        self.model = self.model_class(**self.params)
        self.is_fitted = False
        self.feature_importances_ = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'CLVRegressor':
        """
        Fit the regressor.
        
        Args:
            X: Training features
            y: Training targets
        
        Returns:
            self
        """
        print(f"Training {self.model_type} regressor...")
        self.model.fit(X, y)
        self.is_fitted = True
        
        # Extract feature importances
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importances_ = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            self.feature_importances_ = np.abs(self.model.coef_)
        
        print("Training complete!")
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict target values.
        
        Args:
            X: Features to predict
        
        Returns:
            Predicted values
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted yet!")
        return self.model.predict(X)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Evaluate model performance.
        
        Args:
            X: Test features
            y: True values
        
        Returns:
            Dictionary with evaluation metrics
        """
        y_pred = self.predict(X)
        
        metrics = {
            'rmse': np.sqrt(mean_squared_error(y, y_pred)),
            'mae': mean_absolute_error(y, y_pred),
            'r2': r2_score(y, y_pred)
        }
        
        print("\nRegression Evaluation Metrics:")
        print("-" * 40)
        for metric, value in metrics.items():
            print(f"{metric.upper()}: {value:.4f}")
        
        return metrics
    
    def cross_validate(self, X: np.ndarray, y: np.ndarray, 
                       scoring: str = 'neg_mean_squared_error') -> dict:
        """
        Perform cross-validation.
        
        Args:
            X: Features
            y: Targets
            scoring: Scoring metric
        
        Returns:
            Dictionary with CV results
        """
        print(f"Performing {CV_FOLDS}-fold cross-validation...")
        scores = cross_val_score(self.model, X, y, cv=CV_FOLDS, 
                                  scoring=scoring, n_jobs=N_JOBS)
        
        # Convert negative MSE to positive RMSE
        if 'neg' in scoring:
            scores = np.sqrt(-scores)
            metric_name = 'RMSE'
        else:
            metric_name = scoring
        
        results = {
            'scores': scores,
            'mean': scores.mean(),
            'std': scores.std()
        }
        
        print(f"CV {metric_name}: {results['mean']:.4f} (+/- {results['std']:.4f})")
        return results
    
    def save(self, file_path: Path):
        """Save model to disk."""
        file_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, file_path)
        print(f"Model saved to {file_path}")
    
    @classmethod
    def load(cls, file_path: Path) -> 'CLVRegressor':
        """Load model from disk."""
        model = joblib.load(file_path)
        print(f"Model loaded from {file_path}")
        return model


def predict_clv(df: pd.DataFrame, features: list, 
                target: str = 'TotalCharges') -> pd.DataFrame:
    """
    Predict Customer Lifetime Value using tenure and other features.
    
    Args:
        df: DataFrame with customer data
        features: Feature columns
        target: Target column for CLV
    
    Returns:
        DataFrame with predicted CLV
    """
    # Remove rows where target is invalid
    df_valid = df[df[target].notna()].copy()
    
    X = df_valid[features].values
    y = df_valid[target].values
    
    # Train model
    regressor = CLVRegressor('xgboost')
    regressor.fit(X, y)
    
    # Predict
    df_valid['Predicted_CLV'] = regressor.predict(X)
    
    return df_valid, regressor
