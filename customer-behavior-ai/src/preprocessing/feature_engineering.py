"""
Feature engineering module for Telco Customer Churn dataset
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
import joblib
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from config import (
    PROCESSED_DATA_DIR, TRAIN_DATA_DIR, VAL_DATA_DIR, TEST_DATA_DIR,
    MODELS_DIR, PROCESSED_DATA_FILE, TRAIN_DATA_FILE, VAL_DATA_FILE, TEST_DATA_FILE,
    CATEGORICAL_COLUMNS, NUMERICAL_COLUMNS, TARGET_COLUMN, ID_COLUMN,
    SCALER_FILE, ENCODER_FILE, RANDOM_STATE, TEST_SIZE, VAL_SIZE,
    create_directories
)


def load_processed_data(file_path: Path = None) -> pd.DataFrame:
    """
    Load processed data from CSV file.
    
    Args:
        file_path: Path to processed data file
    
    Returns:
        DataFrame with processed data
    """
    if file_path is None:
        file_path = PROCESSED_DATA_DIR / PROCESSED_DATA_FILE
    
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} rows from {file_path}")
    return df


def create_tenure_groups(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create tenure groups based on customer tenure.
    
    Args:
        df: Input DataFrame
    
    Returns:
        DataFrame with tenure groups
    """
    df = df.copy()
    
    # Tạo nhóm tenure
    bins = [0, 12, 24, 48, 72]
    labels = ['0-12 months', '12-24 months', '24-48 months', '48+ months']
    df['TenureGroup'] = pd.cut(df['tenure'], bins=bins, labels=labels, include_lowest=True)
    
    return df


def create_charges_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create additional features from charges columns.
    
    Args:
        df: Input DataFrame
    
    Returns:
        DataFrame with new charge features
    """
    df = df.copy()
    
    # Average monthly charge (total / tenure)
    df['AvgMonthlyCharge'] = np.where(
        df['tenure'] > 0,
        df['TotalCharges'] / df['tenure'],
        df['MonthlyCharges']
    )
    
    # Charge increase: difference between current monthly and average
    df['ChargeIncrease'] = df['MonthlyCharges'] - df['AvgMonthlyCharge']
    
    # High value customer flag
    df['HighValue'] = (df['MonthlyCharges'] > df['MonthlyCharges'].median()).astype(int)
    
    return df


def create_service_count(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create feature counting total number of services.
    
    Args:
        df: Input DataFrame
    
    Returns:
        DataFrame with service count
    """
    df = df.copy()
    
    service_columns = [
        'PhoneService', 'MultipleLines', 'InternetService', 
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies'
    ]
    
    # Count services (Yes = 1, others = 0)
    service_count = 0
    for col in service_columns:
        if col in df.columns:
            service_count += (df[col] == 'Yes').astype(int)
    
    df['TotalServices'] = service_count
    
    return df


def create_contract_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create contract-related features.
    
    Args:
        df: Input DataFrame
    
    Returns:
        DataFrame with contract features
    """
    df = df.copy()
    
    # Short-term contract flag
    df['ShortTermContract'] = (df['Contract'] == 'Month-to-month').astype(int)
    
    # Automatic payment flag
    df['AutomaticPayment'] = df['PaymentMethod'].apply(
        lambda x: 1 if 'automatic' in str(x).lower() else 0
    )
    
    return df


def encode_categorical_features(df: pd.DataFrame, fit: bool = True, 
                                 encoders: dict = None) -> tuple:
    """
    Encode categorical features using Label Encoding.
    
    Args:
        df: Input DataFrame
        fit: Whether to fit new encoders
        encoders: Pre-fitted encoders (if fit=False)
    
    Returns:
        Tuple of (encoded DataFrame, encoders dict)
    """
    df = df.copy()
    
    if encoders is None:
        encoders = {}
    
    # Cập nhật danh sách categorical columns
    cat_columns = [col for col in CATEGORICAL_COLUMNS if col in df.columns]
    # Thêm SeniorCitizen vì đã convert sang Yes/No
    if 'SeniorCitizen' in df.columns:
        cat_columns.append('SeniorCitizen')
    
    for col in cat_columns:
        if fit:
            le = LabelEncoder()
            df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
        else:
            if col in encoders:
                # Handle unseen labels
                le = encoders[col]
                df[f'{col}_encoded'] = df[col].astype(str).apply(
                    lambda x: le.transform([x])[0] if x in le.classes_ else -1
                )
    
    return df, encoders


def scale_numerical_features(df: pd.DataFrame, fit: bool = True, 
                              scaler: StandardScaler = None) -> tuple:
    """
    Scale numerical features using StandardScaler.
    
    Args:
        df: Input DataFrame
        fit: Whether to fit new scaler
        scaler: Pre-fitted scaler (if fit=False)
    
    Returns:
        Tuple of (scaled DataFrame, scaler)
    """
    df = df.copy()
    
    # Cập nhật danh sách numerical columns
    num_columns = [col for col in NUMERICAL_COLUMNS if col in df.columns]
    # Thêm các features mới
    additional_num = ['AvgMonthlyCharge', 'ChargeIncrease', 'TotalServices']
    num_columns.extend([col for col in additional_num if col in df.columns])
    
    if fit:
        scaler = StandardScaler()
        df_scaled = scaler.fit_transform(df[num_columns])
    else:
        df_scaled = scaler.transform(df[num_columns])
    
    # Create scaled column names
    scaled_columns = [f'{col}_scaled' for col in num_columns]
    df[scaled_columns] = df_scaled
    
    return df, scaler


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Main function to prepare all features.
    
    Args:
        df: Processed DataFrame
    
    Returns:
        DataFrame with engineered features
    """
    print("=" * 50)
    print("Starting Feature Engineering")
    print("=" * 50)
    
    print("\n1. Creating tenure groups...")
    df = create_tenure_groups(df)
    
    print("2. Creating charge features...")
    df = create_charges_features(df)
    
    print("3. Creating service count...")
    df = create_service_count(df)
    
    print("4. Creating contract features...")
    df = create_contract_features(df)
    
    print(f"\nFeature engineering complete! Shape: {df.shape}")
    
    return df


def split_data(df: pd.DataFrame, target_col: str = 'Churn_Binary') -> tuple:
    """
    Split data into train, validation, and test sets.
    
    Args:
        df: Feature-engineered DataFrame
        target_col: Name of target column
    
    Returns:
        Tuple of (train, val, test) DataFrames
    """
    # First split: train+val vs test
    train_val, test = train_test_split(
        df, test_size=TEST_SIZE, random_state=RANDOM_STATE, 
        stratify=df[target_col]
    )
    
    # Second split: train vs val
    val_size_adjusted = VAL_SIZE / (1 - TEST_SIZE)
    train, val = train_test_split(
        train_val, test_size=val_size_adjusted, random_state=RANDOM_STATE,
        stratify=train_val[target_col]
    )
    
    print(f"Train set: {len(train)} samples ({len(train)/len(df)*100:.1f}%)")
    print(f"Validation set: {len(val)} samples ({len(val)/len(df)*100:.1f}%)")
    print(f"Test set: {len(test)} samples ({len(test)/len(df)*100:.1f}%)")
    
    return train, val, test


def get_feature_columns(df: pd.DataFrame) -> list:
    """
    Get list of feature columns for modeling.
    
    Args:
        df: DataFrame with features
    
    Returns:
        List of feature column names
    """
    # Exclude ID, target, and original categorical columns
    exclude_cols = [ID_COLUMN, TARGET_COLUMN, 'Churn_Binary', 'TenureGroup']
    exclude_cols.extend(CATEGORICAL_COLUMNS)
    exclude_cols.append('SeniorCitizen')
    
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    return feature_cols


def save_data_splits(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame):
    """
    Save data splits to CSV files.
    
    Args:
        train: Training DataFrame
        val: Validation DataFrame
        test: Test DataFrame
    """
    create_directories()
    
    train.to_csv(TRAIN_DATA_DIR / TRAIN_DATA_FILE, index=False)
    val.to_csv(VAL_DATA_DIR / VAL_DATA_FILE, index=False)
    test.to_csv(TEST_DATA_DIR / TEST_DATA_FILE, index=False)
    
    print(f"\nSaved data splits:")
    print(f"  Train: {TRAIN_DATA_DIR / TRAIN_DATA_FILE}")
    print(f"  Val: {VAL_DATA_DIR / VAL_DATA_FILE}")
    print(f"  Test: {TEST_DATA_DIR / TEST_DATA_FILE}")


def save_transformers(scaler: StandardScaler, encoders: dict):
    """
    Save fitted transformers for inference.
    
    Args:
        scaler: Fitted StandardScaler
        encoders: Dictionary of fitted LabelEncoders
    """
    create_directories()
    
    joblib.dump(scaler, MODELS_DIR / SCALER_FILE)
    joblib.dump(encoders, MODELS_DIR / ENCODER_FILE)
    
    print(f"\nSaved transformers:")
    print(f"  Scaler: {MODELS_DIR / SCALER_FILE}")
    print(f"  Encoders: {MODELS_DIR / ENCODER_FILE}")


def main():
    """Main execution function."""
    # Create directories
    create_directories()
    
    # Load processed data
    df = load_processed_data()
    
    # Feature engineering
    df = prepare_features(df)
    
    # Encode and scale
    print("\n5. Encoding categorical features...")
    df, encoders = encode_categorical_features(df)
    
    print("6. Scaling numerical features...")
    df, scaler = scale_numerical_features(df)
    
    # Split data
    print("\n7. Splitting data...")
    train, val, test = split_data(df)
    
    # Save everything
    save_data_splits(train, val, test)
    save_transformers(scaler, encoders)
    
    print("\n" + "=" * 50)
    print("Feature Engineering Complete!")
    print("=" * 50)
    
    return train, val, test


if __name__ == "__main__":
    train, val, test = main()
