"""
Inference package
"""
from .predict import ChurnPredictor, predict_from_csv

__all__ = ['ChurnPredictor', 'predict_from_csv']
