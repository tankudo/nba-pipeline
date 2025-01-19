import logging
from typing import Dict, Any
import pandas as pd
import numpy as np
from data.feature_store import FeatureStore
from models.model import NextBestActionModel
from utils.preprocessing import Preprocessor
from monitoring.metrics import MetricsTracker

class PredictionPipeline:
    def __init__(self, config_path: str):
        """
        Initialize the prediction pipeline with necessary components
        
        Args:
            config_path: Path to configuration file
        """
        self.feature_store = FeatureStore(config_path)
        self.model = NextBestActionModel(config_path)
        self.preprocessor = Preprocessor()
        self.metrics_tracker = MetricsTracker()
        self.logger = logging.getLogger(__name__)

    def prepare_features(self, user_id: str, current_event: Dict[str, Any]) -> pd.DataFrame:
        """
        Prepare features by combining historical and real-time data
        
        Args:
            user_id: User identifier
            current_event: Current event data
            
        Returns:
            DataFrame with prepared features
        """
        try:
            # Get historical features from feature store
            historical_features = self.feature_store.get_user_features(user_id)
            
            # Combine with current event
            combined_features = {
                **(historical_features or {}),
                **current_event
            }
            
            # Convert to DataFrame and preprocess
            df = pd.DataFrame([combined_features])
            processed_features = self.preprocessor.preprocess_features(df)
            
            return processed_features
            
        except Exception as e:
            self.logger.error(f"Error preparing features: {str(e)}")
            raise

    def make_prediction(self, features: pd.DataFrame) -> Dict[str, Any]:
        """
        Make prediction using the prepared features
        
        Args:
            features: Prepared feature DataFrame
            
        Returns:
            Dictionary containing prediction results
        """
        try:
            # Get model prediction
            prediction_proba = self.model.predict(features)
            
            # Get the best action and its probability
            best_action_idx = np.argmax(prediction_proba[0])
            confidence = prediction_proba[0][best_action_idx]
            
            # Track prediction metrics
            self.metrics_tracker.track_prediction(
                features=features,
                prediction=best_action_idx,
                confidence=confidence
            )
            
            return {
                'action': int(best_action_idx),
                'confidence': float(confidence),
                'all_probabilities': prediction_proba[0].tolist()
            }
            
        except Exception as e:
            self.logger.error(f"Error making prediction: {str(e)}")
            raise

    def process_event(self, user_id: str, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single event through the pipeline
        
        Args:
            user_id: User identifier
            event_data: Event data
            
        Returns:
            Dictionary containing prediction results
        """
        try:
            # Prepare features
            features = self.prepare_features(user_id, event_data)
            
            # Make prediction
            prediction = self.make_prediction(features)
            
            # Update feature store with new event data
            self.feature_store.update_features(user_id, event_data)
            
            return prediction
            
        except Exception as e:
            self.logger.error(f"Error processing event: {str(e)}")
            raise