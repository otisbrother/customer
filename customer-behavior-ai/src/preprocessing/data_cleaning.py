"""
Data cleaning module for Telco Customer Churn dataset
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from config import (
    RAW_DATA_DIR, PROCESSED_DATA_DIR, RAW_DATA_FILE, PROCESSED_DATA_FILE,
    CATEGORICAL_COLUMNS, NUMERICAL_COLUMNS, TARGET_COLUMN, ID_COLUMN,
    create_directories
)


def load_raw_data(file_path: Path = None) -> pd.DataFrame:
    """
    Load raw data from CSV file.
    
    Args:
        file_path: Path to raw data file. If None, uses default path.
    
    Returns:
        DataFrame with raw data
    """
    if file_path is None:
        file_path = RAW_DATA_DIR / RAW_DATA_FILE
    
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} rows from {file_path}")
    return df


def check_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Check and report missing values in the dataset.
    
    Args:
        df: Input DataFrame
    
    Returns:
        DataFrame with missing value statistics
    """
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    
    missing_df = pd.DataFrame({
        'Column': df.columns,
        'Missing Count': missing.values,
        'Missing Percentage': missing_pct.values
    })
    
    missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values(
        by='Missing Count', ascending=False
    )
    
    if len(missing_df) > 0:
        print("\nMissing Values Found:")
        print(missing_df.to_string(index=False))
    else:
        print("\nNo missing values found!")
    
    return missing_df


def clean_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean TotalCharges column - convert to numeric and handle missing values.
    
    Args:
        df: Input DataFrame
    
    Returns:
        DataFrame with cleaned TotalCharges
    """
    df = df.copy()
    
    # TotalCharges có thể chứa whitespace thay vì số
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    # Fill missing TotalCharges with MonthlyCharges * tenure (nếu tenure = 0, set = MonthlyCharges)
    mask = df['TotalCharges'].isnull()
    if mask.any():
        print(f"Found {mask.sum()} missing TotalCharges values")
        df.loc[mask, 'TotalCharges'] = df.loc[mask, 'MonthlyCharges']
    
    return df


def clean_senior_citizen(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert SeniorCitizen from 0/1 to No/Yes for consistency.
    
    Args:
        df: Input DataFrame
    
    Returns:
        DataFrame with converted SeniorCitizen
    """
    df = df.copy()
    df['SeniorCitizen'] = df['SeniorCitizen'].map({0: 'No', 1: 'Yes'})
    return df


def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode target variable (Churn) to binary.
    
    Args:
        df: Input DataFrame
    
    Returns:
        DataFrame with encoded target
    """
    df = df.copy()
    df['Churn_Binary'] = df[TARGET_COLUMN].map({'No': 0, 'Yes': 1})
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate rows from DataFrame.
    
    Args:
        df: Input DataFrame
    
    Returns:
        DataFrame without duplicates
    """
    initial_len = len(df)
    df = df.drop_duplicates()
    removed = initial_len - len(df)
    
    if removed > 0:
        print(f"Removed {removed} duplicate rows")
    else:
        print("No duplicate rows found")
    
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Main function to clean the entire dataset.
    
    Args:
        df: Raw input DataFrame
    
    Returns:
        Cleaned DataFrame
    """
    print("=" * 50)
    print("Starting Data Cleaning Process")
    print("=" * 50)
    
    # Check initial state
    print(f"\nInitial shape: {df.shape}")
    check_missing_values(df)
    
    # Apply cleaning steps
    print("\n1. Cleaning TotalCharges column...")
    df = clean_total_charges(df)
    
    print("\n2. Converting SeniorCitizen to categorical...")
    df = clean_senior_citizen(df)
    
    print("\n3. Encoding target variable...")
    df = encode_target(df)
    
    print("\n4. Removing duplicates...")
    df = remove_duplicates(df)
    
    # Final check
    print(f"\nFinal shape: {df.shape}")
    check_missing_values(df)
    
    print("\n" + "=" * 50)
    print("Data Cleaning Complete!")
    print("=" * 50)
    
    return df


def save_processed_data(df: pd.DataFrame, file_path: Path = None) -> None:
    """
    Save processed data to CSV file.
    
    Args:
        df: Processed DataFrame
        file_path: Output path. If None, uses default path.
    """
    if file_path is None:
        file_path = PROCESSED_DATA_DIR / PROCESSED_DATA_FILE
    
    # Ensure directory exists
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(file_path, index=False)
    print(f"Saved processed data to {file_path}")


def main():
    """Main execution function."""
    # Create directories
    create_directories()
    
    # Load raw data
    df = load_raw_data()
    
    # Clean data
    df_clean = clean_data(df)
    
    # Save processed data
    save_processed_data(df_clean)
    
    return df_clean


if __name__ == "__main__":
    df = main()
