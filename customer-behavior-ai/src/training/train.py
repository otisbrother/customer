"""
Training script for Customer Churn Prediction models
"""
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import sys
import warnings

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from config import (
    TRAIN_DATA_DIR, VAL_DATA_DIR, TEST_DATA_DIR, MODELS_DIR,
    TRAIN_DATA_FILE, VAL_DATA_FILE, TEST_DATA_FILE,
    CHURN_MODEL_FILE, CLUSTERING_MODEL_FILE,
    RANDOM_STATE, create_directories
)
from models.classification import ChurnClassifier, compare_models
from models.clustering import CustomerSegmentation, analyze_segments
from evaluation.metrics import evaluate_classification, plot_evaluation_results
from preprocessing.feature_engineering import get_feature_columns

warnings.filterwarnings('ignore')


def load_data():
    """Load train, validation, and test datasets."""
    train = pd.read_csv(TRAIN_DATA_DIR / TRAIN_DATA_FILE)
    val = pd.read_csv(VAL_DATA_DIR / VAL_DATA_FILE)
    test = pd.read_csv(TEST_DATA_DIR / TEST_DATA_FILE)
    
    print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    return train, val, test


def prepare_features(df: pd.DataFrame) -> tuple:
    """
    Prepare feature matrix and target vector.
    
    Args:
        df: DataFrame with features
    
    Returns:
        Tuple of (X, y, feature_names)
    """
    # Get scaled and encoded features
    feature_cols = [col for col in df.columns if 
                    col.endswith('_scaled') or col.endswith('_encoded') or
                    col in ['HighValue', 'ShortTermContract', 'AutomaticPayment', 'TotalServices']]
    
    X = df[feature_cols].values
    y = df['Churn_Binary'].values
    
    return X, y, feature_cols


def train_churn_model(train: pd.DataFrame, val: pd.DataFrame,
                      model_type: str = 'xgboost',
                      tune: bool = False) -> ChurnClassifier:
    """
    Train a churn prediction model.
    
    Args:
        train: Training data
        val: Validation data
        model_type: Type of classifier
        tune: Whether to tune hyperparameters
    
    Returns:
        Trained ChurnClassifier
    """
    print("\n" + "=" * 60)
    print(f"Training {model_type.upper()} Churn Prediction Model")
    print("=" * 60)
    
    # Prepare data
    X_train, y_train, feature_cols = prepare_features(train)
    X_val, y_val, _ = prepare_features(val)
    
    print(f"\nFeatures: {len(feature_cols)}")
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Churn rate (train): {y_train.mean():.2%}")
    
    # Create and train model
    model = ChurnClassifier(model_type=model_type)
    
    if tune:
        print("\nTuning hyperparameters...")
        model.tune_hyperparameters(X_train, y_train)
    else:
        model.fit(X_train, y_train)
    
    # Cross-validation on training data
    print("\nCross-validation results:")
    model.cross_validate(X_train, y_train, scoring='roc_auc')
    
    # Validation evaluation
    print("\nValidation set evaluation:")
    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)[:, 1]
    
    metrics = evaluate_classification(y_val, y_pred, y_proba)
    
    # Feature importance
    print("\nTop 10 Feature Importances:")
    importance_df = model.get_feature_importance(feature_cols)
    if importance_df is not None:
        print(importance_df.head(10).to_string(index=False))
    
    return model, metrics, feature_cols


def train_clustering_model(df: pd.DataFrame, n_clusters: int = 4) -> CustomerSegmentation:
    """
    Train a customer segmentation model.
    
    Args:
        df: DataFrame with customer data
        n_clusters: Number of clusters
    
    Returns:
        Trained CustomerSegmentation model
    """
    print("\n" + "=" * 60)
    print("Training Customer Segmentation Model")
    print("=" * 60)
    
    # Select numerical features for clustering
    cluster_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
    scaled_features = [f'{col}_scaled' for col in cluster_features]
    
    # Check which features exist
    available_features = [col for col in scaled_features if col in df.columns]
    if not available_features:
        available_features = cluster_features
    
    print(f"\nClustering features: {available_features}")
    
    X = df[available_features].dropna().values
    
    # Find optimal k
    model = CustomerSegmentation('kmeans')
    optimal_results = model.find_optimal_k(X, k_range=range(2, 8))
    
    # Train with specified or optimal k
    model = CustomerSegmentation('kmeans', n_clusters=n_clusters)
    model.fit(X)
    
    # Evaluate
    metrics = model.evaluate(X)
    
    return model, metrics


def train_all_models(tune: bool = False):
    """
    Train all models (churn classifier and customer segmentation).
    
    Args:
        tune: Whether to tune hyperparameters
    """
    create_directories()
    
    # Load data
    print("\nLoading data...")
    train, val, test = load_data()
    
    # Compare different classifiers
    print("\n" + "=" * 60)
    print("COMPARING CLASSIFICATION MODELS")
    print("=" * 60)
    
    X_train, y_train, feature_cols = prepare_features(train)
    X_test, y_test, _ = prepare_features(test)
    
    comparison_results = compare_models(X_train, y_train, X_test, y_test)
    
    # Train best model (XGBoost typically performs best)
    churn_model, churn_metrics, _ = train_churn_model(train, val, 'xgboost', tune)
    
    # Save churn model
    churn_model.save(MODELS_DIR / CHURN_MODEL_FILE)
    
    # Train clustering model
    clustering_model, cluster_metrics = train_clustering_model(train)
    
    # Save clustering model
    clustering_model.save(MODELS_DIR / CLUSTERING_MODEL_FILE)
    
    # Final test evaluation
    print("\n" + "=" * 60)
    print("FINAL TEST SET EVALUATION")
    print("=" * 60)
    
    y_pred_test = churn_model.predict(X_test)
    y_proba_test = churn_model.predict_proba(X_test)[:, 1]
    
    test_metrics = evaluate_classification(y_test, y_pred_test, y_proba_test)
    
    # Plot evaluation results
    plot_evaluation_results(y_test, y_pred_test, y_proba_test, 
                            save_path=Path('reports/figures'))
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"\nModels saved to: {MODELS_DIR}")
    print(f"  - Churn model: {CHURN_MODEL_FILE}")
    print(f"  - Clustering model: {CLUSTERING_MODEL_FILE}")
    
    return {
        'comparison': comparison_results,
        'churn_metrics': churn_metrics,
        'cluster_metrics': cluster_metrics,
        'test_metrics': test_metrics
    }


def main():
    """Main training script."""
    parser = argparse.ArgumentParser(description='Train Customer Churn Models')
    parser.add_argument('--model', type=str, default='xgboost',
                       choices=['logistic', 'random_forest', 'xgboost', 'gradient_boosting'],
                       help='Type of classification model')
    parser.add_argument('--tune', action='store_true',
                       help='Tune hyperparameters')
    parser.add_argument('--compare', action='store_true',
                       help='Compare multiple models')
    
    args = parser.parse_args()
    
    if args.compare:
        results = train_all_models(tune=args.tune)
    else:
        train, val, test = load_data()
        model, metrics, _ = train_churn_model(train, val, args.model, args.tune)
        model.save()


if __name__ == "__main__":
    main()
