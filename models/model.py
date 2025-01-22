import mlflow
import mlflow.lightgbm
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import yaml

class NextBestActionModel:
    def __init__(self, config_path: str, model_name: str = "nba_model"):
        """
        Initialize the Next Best Action Model
        
        Args:
            config_path: Path to configuration file
            model_name: Name of the model for MLflow tracking
        """
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
            
        self.model_name = model_name
        self.model = None
        self.model_uri = None
        self.feature_encoders = {}
        self.scaler = StandardScaler()

        # Default model parameters optimized for NBA use case
        self.model_params = {
            'objective': 'multiclass',
            'num_class': 5, 
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'max_depth': 6,
            'min_data_in_leaf': 50,
            'verbose': -1
        }
        
        # Set up MLflow tracking
        mlflow.set_tracking_uri("file:./mlruns")

    def _get_latest_model(self) -> Optional[str]:
        """Get the URI of the latest model version from MLflow"""
        client = mlflow.tracking.MlflowClient()
        try:
            latest_version = client.get_latest_versions(self.model_name, stages=["Production"])[0]
            return latest_version.source
        except:
            return None

    def load_model(self) -> bool:
        """Load the latest model version from MLflow"""
        model_uri = self._get_latest_model()
        if model_uri:
            self.model = mlflow.lightgbm.load_model(model_uri)
            self.model_uri = model_uri
            return True
        return False

    def preprocess_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Comprehensive feature preprocessing pipeline.
        """
        # Handle missing values
        df = self._handle_missing_values(df)
        
        # Create temporal features
        df = self._create_temporal_features(df)
        
        # Create interaction features
        df = self._create_interaction_features(df)
        
        # Encode categorical variables
        df = self._encode_categorical_features(df)
        
        # Scale numerical features
        numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
        df[numerical_cols] = self.scaler.fit_transform(df[numerical_cols])
        
        return df

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handles missing values by using median for skewed distributions and mean for normal distributions.
        """
        numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
        for col in numerical_cols:
            # Use median for skewed distributions
            if df[col].skew() > 1:
                df[col].fillna(df[col].median(), inplace=True)
            else:
                # Use mean for normal distributions
                df[col].fillna(df[col].mean(), inplace=True)
            # Add missing indicator column
            df[f'{col}_is_missing'] = df[col].isna().astype(int)
        
        # For categorical columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            df[col].fillna('MISSING', inplace=True)
            
        return df

    def _create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create time-based features like hour of day, day of week, etc.
        """
        if 'timestamp' in df.columns:
            df['hour_of_day'] = pd.to_datetime(df['timestamp']).dt.hour
            df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
            df['month'] = pd.to_datetime(df['timestamp']).dt.month
            # Time since last action
            df['time_since_last_action'] = df.groupby('user_id')['timestamp'].diff().dt.total_seconds()
        
        return df

    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features between key variables.
        """
        if 'amount' in df.columns and 'frequency' in df.columns:
            df['amount_per_frequency'] = df['amount'] / (df['frequency'] + 1)
        
        if 'age' in df.columns and 'income' in df.columns:
            df['income_per_age'] = df['income'] / (df['age'] + 1)
            
        return df

    def _encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical features using label encoding.
        """
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col not in self.feature_encoders:
                encoder = LabelEncoder()
                df[col] = encoder.fit_transform(df[col].astype(str))
                self.feature_encoders[col] = encoder
            else:
                df[col] = self.feature_encoders[col].transform(df[col].astype(str))
        
        return df

    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """
        Train the model with hyperparameter optimization and MLflow tracking.
        """
        with mlflow.start_run():
            X_processed = self.preprocess_features(X_train)
            train_data = lgb.Dataset(X_processed, label=y_train)
            
            # Hyperparameter optimization
            param_grid = {
                'num_leaves': [15, 31, 63],
                'learning_rate': [0.01, 0.05, 0.1],
                'feature_fraction': [0.7, 0.8, 0.9],
                'bagging_fraction': [0.7, 0.8, 0.9],
                'max_depth': [4, 6, 8],
                'min_data_in_leaf': [30, 50, 100]
            }
            
            # Random search for hyperparameter optimization
            random_search = RandomizedSearchCV(
                estimator=lgb.LGBMClassifier(),
                param_distributions=param_grid,
                n_iter=20,
                cv=5,
                random_state=42
            )
            
            random_search.fit(X_processed, y_train)
            
            # Update model parameters with best found parameters
            self.model_params.update(random_search.best_params_)
            
            # Log parameters
            mlflow.log_params(self.model_params)
            
            # Train final model
            self.model = lgb.train(
                self.model_params,
                train_data,
                num_boost_round=1000,
                early_stopping_rounds=50
            )
            
            # Log metrics
            mlflow.log_metric("best_iteration", self.model.best_iteration)
            
            # Log model
            mlflow.lightgbm.log_model(
                self.model,
                artifact_path=self.model_name,
                registered_model_name=self.model_name
            )
            
            # Transition model to production
            client = mlflow.tracking.MlflowClient()
            latest_version = client.get_latest_versions(self.model_name, stages=["None"])[0]
            client.transition_model_version_stage(
                name=self.model_name,
                version=latest_version.version,
                stage="Production"
            )
            
            self.model_uri = latest_version.source

    def predict_proba_realtime(self, features: Dict) -> np.ndarray:
        """
        Real-time prediction with feature preprocessing.
        """
        if self.model is None:
            if not self.load_model():
                raise ValueError("No trained model available. Please train the model first.")
                
        # Convert single instance to DataFrame
        df = pd.DataFrame([features])
        
        # Apply preprocessing
        df = self.preprocess_features(df)
        
        # Make prediction
        return self.model.predict(df, num_iteration=self.model.best_iteration)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the trained model
        """
        if self.model is None:
            if not self.load_model():
                raise ValueError("No trained model available. Please train the model first.")
        
        X_processed = self.preprocess_features(X)
        return self.model.predict_proba(X_processed)

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance analysis.
        """
        if self.model is None:
            if not self.load_model():
                raise ValueError("No trained model available. Please train the model first.")
                
        importance_df = pd.DataFrame({
            'feature': self.model.feature_name(),
            'importance': self.model.feature_importance('gain')
        })
        return importance_df.sort_values('importance', ascending=False)
