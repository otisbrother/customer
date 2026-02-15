"""
Customer Behavior AI - Source Package
"""
from . import preprocessing
from . import models
from . import training
from . import evaluation
from . import inference
from . import config

__version__ = '1.0.0'
__author__ = 'Customer Behavior AI Team'

__all__ = [
    'preprocessing',
    'models', 
    'training',
    'evaluation',
    'inference',
    'config'
]
