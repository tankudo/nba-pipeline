import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class Preprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
    
    def preprocess_features(self, df):
        """Preprocess features for model input"""
        # Handle missing values
        df = df.fillna(df.mean())
        
        # Feature engineering
        df['recency'] = (pd.Timestamp.now() - pd.to_datetime(df['last_activity'])).dt.days
        
        # Scale numerical features
        numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
        df[numerical_cols] = self.scaler.fit_transform(df[numerical_cols])
        
        return df