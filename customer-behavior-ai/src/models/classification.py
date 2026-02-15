"""
Classification models for Customer Churn Prediction
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
import joblib
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from config import MODELS_DIR, CHURN_MODEL_FILE, RANDOM_STATE, CV_FOLDS, N_JOBS


class ChurnClassifier:
    """
    Unified classifier class for churn prediction.
    Supports multiple algorithms: LogisticRegression, RandomForest, XGBoost, etc.
    """
    
    SUPPORTED_MODELS = {
        'logistic': LogisticRegression,
        'random_forest': RandomForestClassifier,
        'xgboost': XGBClassifier,
        'gradient_boosting': GradientBoostingClassifier,
        'svm': SVC
    }
    
    DEFAULT_PARAMS = {
        'logistic': {
            'random_state': RANDOM_STATE,
            'max_iter': 1000,
            'class_weight': 'balanced'
        },
        'random_forest': {
            'random_state': RANDOM_STATE,
            'n_estimators': 100,
            'max_depth': 10,
            'class_weight': 'balanced',
            'n_jobs': N_JOBS
        },
        'xgboost': {
            'random_state': RANDOM_STATE,
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'scale_pos_weight': 1,
            'use_label_encoder': False,
            'eval_metric': 'logloss'
        },
        'gradient_boosting': {
            'random_state': RANDOM_STATE,
            'n_estimators': 100,
            'max_depth': 5,
            'learning_rate': 0.1
        },
        'svm': {
            'random_state': RANDOM_STATE,
            'kernel': 'rbf',
            'probability': True,
            'class_weight': 'balanced'
        }
    }
    
    TUNING_PARAMS = {
        'logistic': {
            'C': [0.001, 0.01, 0.1, 1, 10],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga']
        },
        'random_forest': {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        },
        'xgboost': {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        },
        'gradient_boosting': {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2]
        },
        'svm': {
            'C': [0.1, 1, 10],
            'gamma': ['scale', 'auto', 0.1, 0.01]
        }
    }
    
    def __init__(self, model_type: str = 'xgboost', **kwargs):
        """
        Initialize classifier.
        
        Args:
            model_type: Type of classifier ('logistic', 'random_forest', 'xgboost', etc.)
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
        self.feature_importances_ = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'ChurnClassifier':
        """
        Fit the classifier.
        
        Args:
            X: Training features
            y: Training labels
        
        Returns:
            self
        """
        print(f"Training {self.model_type} classifier...")
        self.model.fit(X, y)
        self.is_fitted = True
        
        # Extract feature importances if available
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importances_ = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            self.feature_importances_ = np.abs(self.model.coef_[0])
        
        print("Training complete!")
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.
        
        Args:
            X: Features to predict
        
        Returns:
            Predicted labels
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted yet!")
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Features to predict
        
        Returns:
            Class probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted yet!")
        return self.model.predict_proba(X)
    
    def cross_validate(self, X: np.ndarray, y: np.ndarray, 
                       scoring: str = 'roc_auc') -> dict:
        """
        Perform cross-validation.
        
        Args:
            X: Features
            y: Labels
            scoring: Scoring metric
        
        Returns:
            Dictionary with CV results
        """
        print(f"Performing {CV_FOLDS}-fold cross-validation...")
        scores = cross_val_score(self.model, X, y, cv=CV_FOLDS, 
                                  scoring=scoring, n_jobs=N_JOBS)
        
        results = {
            'scores': scores,
            'mean': scores.mean(),
            'std': scores.std()
        }
        
        print(f"CV {scoring}: {results['mean']:.4f} (+/- {results['std']:.4f})")
        return results
    
    def tune_hyperparameters(self, X: np.ndarray, y: np.ndarray,
                              param_grid: dict = None, 
                              scoring: str = 'roc_auc') -> 'ChurnClassifier':
        """
        Tune hyperparameters using GridSearchCV.
        
        Args:
            X: Features
            y: Labels
            param_grid: Parameter grid (uses default if None)
            scoring: Scoring metric
        
        Returns:
            self with best parameters
        """
        if param_grid is None:
            param_grid = self.TUNING_PARAMS.get(self.model_type, {})
        
        if not param_grid:
            print("No parameter grid defined for this model type.")
            return self
        
        print(f"Tuning hyperparameters for {self.model_type}...")
        print(f"Parameter grid: {param_grid}")
        
        grid_search = GridSearchCV(
            self.model, param_grid, cv=CV_FOLDS, 
            scoring=scoring, n_jobs=N_JOBS, verbose=1
        )
        grid_search.fit(X, y)
        
        print(f"\nBest parameters: {grid_search.best_params_}")
        print(f"Best {scoring}: {grid_search.best_score_:.4f}")
        
        self.model = grid_search.best_estimator_
        self.params.update(grid_search.best_params_)
        self.is_fitted = True
        
        return self
    
    def get_feature_importance(self, feature_names: list = None) -> pd.DataFrame:
        """
        Get feature importance scores.
        
        Args:
            feature_names: List of feature names
        
        Returns:
            DataFrame with feature importances
        """
        if self.feature_importances_ is None:
            print("Feature importances not available for this model.")
            return None
        
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(len(self.feature_importances_))]
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': self.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def save(self, file_path: Path = None):
        """
        Save the model to disk.
        
        Args:
            file_path: Path to save model
        """
        if file_path is None:
            file_path = MODELS_DIR / CHURN_MODEL_FILE
        
        file_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, file_path)
        print(f"Model saved to {file_path}")
    
    @classmethod
    def load(cls, file_path: Path = None) -> 'ChurnClassifier':
        """
        Load a model from disk.
        
        Args:
            file_path: Path to load model from
        
        Returns:
            Loaded ChurnClassifier instance
        """
        if file_path is None:
            file_path = MODELS_DIR / CHURN_MODEL_FILE
        
        model = joblib.load(file_path)
        print(f"Model loaded from {file_path}")
        return model


def compare_models(X_train: np.ndarray, y_train: np.ndarray,
                   X_test: np.ndarray, y_test: np.ndarray,
                   models: list = None) -> pd.DataFrame:
    """
    Compare multiple classification models.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        models: List of model types to compare
    
    Returns:
        DataFrame with model comparison results
    """
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    
    if models is None:
        models = ['logistic', 'random_forest', 'xgboost', 'gradient_boosting']
    
    results = []
    
    for model_type in models:
        print(f"\n{'='*50}")
        print(f"Training {model_type}...")
        print('='*50)
        
        clf = ChurnClassifier(model_type=model_type)
        clf.fit(X_train, y_train)
        
        # Predictions
        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)[:, 1]
        
        # Metrics
        results.append({
            'Model': model_type,
            'Accuracy': accuracy_score(y_test, y_pred),
            'F1 Score': f1_score(y_test, y_pred),
            'ROC AUC': roc_auc_score(y_test, y_proba)
        })
    
    results_df = pd.DataFrame(results).sort_values('ROC AUC', ascending=False)
    
    print("\n" + "="*50)
    print("Model Comparison Results")
    print("="*50)
    print(results_df.to_string(index=False))
    
    return results_df
