"""
Configuration file for Customer Behavior AI project
"""
import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
TRAIN_DATA_DIR = DATA_DIR / "train"
VAL_DATA_DIR = DATA_DIR / "val"
TEST_DATA_DIR = DATA_DIR / "test"

# Model directories
MODELS_DIR = PROJECT_ROOT / "models_saved"

# Reports directory
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Data file names
RAW_DATA_FILE = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
PROCESSED_DATA_FILE = "telco_churn_processed.csv"
TRAIN_DATA_FILE = "train.csv"
VAL_DATA_FILE = "val.csv"
TEST_DATA_FILE = "test.csv"

# Model file names
CHURN_MODEL_FILE = "churn_model.pkl"
CLUSTERING_MODEL_FILE = "clustering_model.pkl"
SCALER_FILE = "scaler.pkl"
ENCODER_FILE = "label_encoder.pkl"

# Feature columns
CATEGORICAL_COLUMNS = [
    'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
    'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
    'PaperlessBilling', 'PaymentMethod'
]

NUMERICAL_COLUMNS = ['tenure', 'MonthlyCharges', 'TotalCharges']

TARGET_COLUMN = 'Churn'
ID_COLUMN = 'customerID'

# Model parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.1

# Training parameters
CV_FOLDS = 5
N_JOBS = -1

# Clustering parameters
N_CLUSTERS = 4

# Create directories if they don't exist
def create_directories():
    """Create all necessary directories for the project."""
    directories = [
        RAW_DATA_DIR, PROCESSED_DATA_DIR, TRAIN_DATA_DIR, VAL_DATA_DIR,
        TEST_DATA_DIR, MODELS_DIR, FIGURES_DIR
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    create_directories()
    print("All directories created successfully!")
