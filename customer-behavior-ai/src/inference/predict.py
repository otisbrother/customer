"""
Prediction/Inference module for Customer Churn
"""
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import argparse
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from config import (
    MODELS_DIR, CHURN_MODEL_FILE, CLUSTERING_MODEL_FILE,
    SCALER_FILE, ENCODER_FILE, CATEGORICAL_COLUMNS
)
from models.classification import ChurnClassifier
from models.clustering import CustomerSegmentation


class ChurnPredictor:
    """
    Inference class for customer churn prediction.
    Handles data preprocessing and prediction in one interface.
    """
    
    def __init__(self, model_path: Path = None):
        """
        Initialize predictor with trained model.
        
        Args:
            model_path: Path to saved model file
        """
        if model_path is None:
            model_path = MODELS_DIR / CHURN_MODEL_FILE
        
        self.model = ChurnClassifier.load(model_path)
        self.scaler = self._load_scaler()
        self.encoders = self._load_encoders()
        
        print("ChurnPredictor initialized successfully!")
    
    def _load_scaler(self):
        """Load fitted scaler."""
        scaler_path = MODELS_DIR / SCALER_FILE
        if scaler_path.exists():
            return joblib.load(scaler_path)
        return None
    
    def _load_encoders(self):
        """Load fitted encoders."""
        encoder_path = MODELS_DIR / ENCODER_FILE
        if encoder_path.exists():
            return joblib.load(encoder_path)
        return None
    
    def preprocess_single(self, customer_data: dict) -> np.ndarray:
        """
        Preprocess a single customer record.
        
        Args:
            customer_data: Dictionary with customer features
        
        Returns:
            Preprocessed feature array
        """
        df = pd.DataFrame([customer_data])
        return self.preprocess_batch(df)
    
    def preprocess_batch(self, df: pd.DataFrame) -> np.ndarray:
        """
        Preprocess a batch of customer records.
        
        Args:
            df: DataFrame with customer features
        
        Returns:
            Preprocessed feature array
        """
        df = df.copy()
        
        # Handle TotalCharges
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df['TotalCharges'].fillna(df['MonthlyCharges'], inplace=True)
        
        # Convert SeniorCitizen if numeric
        if df['SeniorCitizen'].dtype in ['int64', 'float64']:
            df['SeniorCitizen'] = df['SeniorCitizen'].map({0: 'No', 1: 'Yes'})
        
        # Create engineered features
        df['AvgMonthlyCharge'] = np.where(
            df['tenure'] > 0,
            df['TotalCharges'] / df['tenure'],
            df['MonthlyCharges']
        )
        df['ChargeIncrease'] = df['MonthlyCharges'] - df['AvgMonthlyCharge']
        df['HighValue'] = (df['MonthlyCharges'] > 70).astype(int)  # Approximate median
        df['ShortTermContract'] = (df['Contract'] == 'Month-to-month').astype(int)
        df['AutomaticPayment'] = df['PaymentMethod'].apply(
            lambda x: 1 if 'automatic' in str(x).lower() else 0
        )
        
        # Service count
        service_columns = [
            'PhoneService', 'MultipleLines', 'InternetService', 
            'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
            'TechSupport', 'StreamingTV', 'StreamingMovies'
        ]
        df['TotalServices'] = sum((df[col] == 'Yes').astype(int) 
                                   for col in service_columns if col in df.columns)
        
        # Encode categorical
        if self.encoders:
            cat_columns = [col for col in CATEGORICAL_COLUMNS if col in df.columns]
            if 'SeniorCitizen' in df.columns:
                cat_columns.append('SeniorCitizen')
            
            for col in cat_columns:
                if col in self.encoders:
                    le = self.encoders[col]
                    df[f'{col}_encoded'] = df[col].astype(str).apply(
                        lambda x: le.transform([x])[0] if x in le.classes_ else -1
                    )
        
        # Scale numerical
        numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 
                          'AvgMonthlyCharge', 'ChargeIncrease', 'TotalServices']
        available_num_cols = [col for col in numerical_cols if col in df.columns]
        
        if self.scaler and available_num_cols:
            df_scaled = self.scaler.transform(df[available_num_cols])
            scaled_cols = [f'{col}_scaled' for col in available_num_cols]
            df[scaled_cols] = df_scaled
        
        # Get feature columns
        feature_cols = [col for col in df.columns if 
                        col.endswith('_scaled') or col.endswith('_encoded') or
                        col in ['HighValue', 'ShortTermContract', 'AutomaticPayment', 'TotalServices']]
        
        return df[feature_cols].values
    
    def predict(self, customer_data: dict) -> dict:
        """
        Predict churn for a single customer.
        
        Args:
            customer_data: Dictionary with customer features
        
        Returns:
            Dictionary with prediction results
        """
        X = self.preprocess_single(customer_data)
        
        prediction = self.model.predict(X)[0]
        probability = self.model.predict_proba(X)[0]
        
        result = {
            'churn_prediction': 'Yes' if prediction == 1 else 'No',
            'churn_probability': float(probability[1]),
            'retention_probability': float(probability[0]),
            'risk_level': self._get_risk_level(probability[1])
        }
        
        return result
    
    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict churn for multiple customers.
        
        Args:
            df: DataFrame with customer features
        
        Returns:
            DataFrame with predictions
        """
        X = self.preprocess_batch(df)
        
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)
        
        results = df.copy()
        results['Churn_Prediction'] = ['Yes' if p == 1 else 'No' for p in predictions]
        results['Churn_Probability'] = probabilities[:, 1]
        results['Retention_Probability'] = probabilities[:, 0]
        results['Risk_Level'] = [self._get_risk_level(p) for p in probabilities[:, 1]]
        
        return results
    
    def _get_risk_level(self, probability: float) -> str:
        """
        Categorize churn probability into risk levels.
        
        Args:
            probability: Churn probability
        
        Returns:
            Risk level string
        """
        if probability >= 0.7:
            return 'High'
        elif probability >= 0.4:
            return 'Medium'
        else:
            return 'Low'
    
    def get_retention_recommendations(self, customer_data: dict) -> list:
        """
        Generate retention recommendations based on customer profile.
        
        Args:
            customer_data: Dictionary with customer features
        
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        # Contract-based recommendations
        if customer_data.get('Contract') == 'Month-to-month':
            recommendations.append(
                "Offer incentives to upgrade to annual contract (10-20% discount)"
            )
        
        # Payment method recommendations
        if customer_data.get('PaymentMethod') == 'Electronic check':
            recommendations.append(
                "Encourage switch to automatic payment for convenience"
            )
        
        # Service-based recommendations
        if customer_data.get('InternetService') == 'Fiber optic':
            if customer_data.get('TechSupport') == 'No':
                recommendations.append(
                    "Offer free tech support trial (fiber customers benefit greatly)"
                )
            if customer_data.get('OnlineSecurity') == 'No':
                recommendations.append(
                    "Bundle online security at discounted rate"
                )
        
        # Tenure-based recommendations
        tenure = customer_data.get('tenure', 0)
        if tenure < 6:
            recommendations.append(
                "Assign dedicated account manager for new customer onboarding"
            )
        elif tenure > 48:
            recommendations.append(
                "Recognize loyalty with exclusive benefits/rewards"
            )
        
        # High charges recommendations
        monthly_charges = customer_data.get('MonthlyCharges', 0)
        if monthly_charges > 90:
            recommendations.append(
                "Review current plan for optimization opportunities"
            )
        
        return recommendations if recommendations else ["Customer profile looks stable"]


def predict_from_csv(input_file: str, output_file: str = None):
    """
    Make predictions from CSV file.
    
    Args:
        input_file: Path to input CSV
        output_file: Path to output CSV (optional)
    """
    print(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file)
    
    print("Initializing predictor...")
    predictor = ChurnPredictor()
    
    print("Making predictions...")
    results = predictor.predict_batch(df)
    
    # Summary statistics
    print("\n" + "=" * 50)
    print("Prediction Summary")
    print("=" * 50)
    print(f"Total customers: {len(results)}")
    print(f"Predicted churners: {(results['Churn_Prediction'] == 'Yes').sum()}")
    print(f"Predicted churn rate: {(results['Churn_Prediction'] == 'Yes').mean():.2%}")
    
    risk_counts = results['Risk_Level'].value_counts()
    print("\nRisk Distribution:")
    for level in ['High', 'Medium', 'Low']:
        if level in risk_counts.index:
            print(f"  {level}: {risk_counts[level]} ({risk_counts[level]/len(results)*100:.1f}%)")
    
    # Save results
    if output_file:
        results.to_csv(output_file, index=False)
        print(f"\nResults saved to {output_file}")
    
    return results


def main():
    """Main inference script."""
    parser = argparse.ArgumentParser(description='Predict Customer Churn')
    parser.add_argument('--input', type=str, required=True,
                       help='Input CSV file path')
    parser.add_argument('--output', type=str, default=None,
                       help='Output CSV file path (optional)')
    
    args = parser.parse_args()
    
    predict_from_csv(args.input, args.output)


if __name__ == "__main__":
    main()
