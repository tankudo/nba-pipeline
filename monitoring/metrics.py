import logging
from typing import Dict, Any
import pandas as pd
import numpy as np
from datetime import datetime
import json

class MetricsTracker:
    def __init__(self):
        """Initialize the metrics tracker"""
        self.logger = logging.getLogger(__name__)
        self.metrics_buffer = []
        self.buffer_size = 1000

    def track_prediction(self, features: pd.DataFrame, prediction: int, confidence: float) -> None:
        """
        Track metrics for a single prediction
        
        Args:
            features: Input features
            prediction: Predicted action
            confidence: Prediction confidence
        """
        try:
            metrics = {
                'timestamp': datetime.now().isoformat(),
                'prediction': prediction,
                'confidence': confidence,
                'feature_statistics': self._calculate_feature_statistics(features)
            }
            
            self.metrics_buffer.append(metrics)
            
            # Flush buffer if it reaches the size limit
            if len(self.metrics_buffer) >= self.buffer_size:
                self.flush_metrics()
                
        except Exception as e:
            self.logger.error(f"Error tracking metrics: {str(e)}")

    def _calculate_feature_statistics(self, features: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate basic statistics for input features
        
        Args:
            features: Input features
            
        Returns:
            Dictionary containing feature statistics
        """
        try:
            return {
                'mean': features.mean().to_dict(),
                'std': features.std().to_dict(),
                'missing_values': features.isnull().sum().to_dict()
            }
        except Exception as e:
            self.logger.error(f"Error calculating feature statistics: {str(e)}")
            return {}

    def track_feedback(self, prediction_id: str, actual_outcome: int, 
                      user_interaction: Dict[str, Any]) -> None:
        """
        Track feedback received for a prediction
        
        Args:
            prediction_id: Unique identifier for the prediction
            actual_outcome: Actual outcome/action taken
            user_interaction: Dictionary containing user interaction data
        """
        try:
            feedback = {
                'timestamp': datetime.now().isoformat(),
                'prediction_id': prediction_id,
                'actual_outcome': actual_outcome,
                'user_interaction': user_interaction
            }
            
            self._store_feedback(feedback)
            
        except Exception as e:
            self.logger.error(f"Error tracking feedback: {str(e)}")

    def _store_feedback(self, feedback: Dict[str, Any]) -> None:
        """
        Store feedback data
        
        Args:
            feedback: Feedback data to store
        """
        try:
            # In a real implementation, this would store to a database
            # For now, we'll just log it
            self.logger.info(f"Feedback received: {json.dumps(feedback)}")
        except Exception as e:
            self.logger.error(f"Error storing feedback: {str(e)}")

    def flush_metrics(self) -> None:
        """Flush metrics buffer to storage"""
        try:
            if self.metrics_buffer:
                # In a real implementation, this would write to a database or monitoring service
                self.logger.info(f"Flushing {len(self.metrics_buffer)} metrics records")
                self.metrics_buffer = []
        except Exception as e:
            self.logger.error(f"Error flushing metrics: {str(e)}")

    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Get summary of tracked metrics
        
        Returns:
            Dictionary containing metrics summary
        """
        try:
            if not self.metrics_buffer:
                return {}
                
            df = pd.DataFrame(self.metrics_buffer)
            
            return {
                'total_predictions': len(df),
                'average_confidence': float(df['confidence'].mean()),
                'confidence_std': float(df['confidence'].std()),
                'prediction_distribution': df['prediction'].value_counts().to_dict()
            }
            
        except Exception as e:
            self.logger.error(f"Error generating metrics summary: {str(e)}")
            return {}