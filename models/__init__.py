"""
Machine Learning Models Package for Next Best Action System

This package contains model-related components including:
- Base model class definitions
- Model training utilities
- Model versioning and serialization
- Prediction interfaces

Classes:
    NextBestActionModel: Main model class for NBA predictions
    ModelTrainer: Utility class for model training
    ModelRegistry: Handles model versioning and storage

Usage:
    from models import NextBestActionModel
    model = NextBestActionModel(config_path='config.yaml')
    predictions = model.predict(features)
"""

from .model import NextBestActionModel
# from .train import ModelTrainer
# Import ModelRegistry if it's defined in another file, e.g.:
# from .model_registry import ModelRegistry

__version__ = '1.0.0'

__all__ = [
    'NextBestActionModel',
#     'ModelTrainer',
    # Add ModelRegistry if it's implemented elsewhere
    # 'ModelRegistry'
]