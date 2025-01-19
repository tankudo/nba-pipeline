from data.data_ingestion import DataIngestion
from data.feature_store import FeatureStore
from models.model import NextBestActionModel
from utils.preprocessing import Preprocessor
import mlflow
from data.download_data import download_kaggle_data

def main():
    
    # Download the Kaggle dataset and load it into a DataFrame
    dataset_name = "drgilermo/nba-players-stats"  # Example Kaggle dataset
    historical_data = download_kaggle_data(dataset_name)
    
    if historical_data is None:
        print("Failed to load dataset. Exiting...")
        return
    
    # Initialize components
    data_ingestion = DataIngestion("config/config.yaml")
    feature_store = FeatureStore("config/config.yaml")
    model = NextBestActionModel("config/config.yaml")
    preprocessor = Preprocessor()
    
    # Preprocess historical data
    processed_data = preprocessor.preprocess_features(historical_data)
    
    # Train model
    X = processed_data.drop('target', axis=1)
    y = processed_data['target']
    model.train(X, y)
    
    # Start real-time prediction pipeline
    for event in data_ingestion.stream_events():
        # Update feature store
        feature_store.update_features(event['user_id'], event['features'])
        
        # Make prediction
        features = preprocessor.preprocess_features(pd.DataFrame([event['features']]))
        prediction = model.predict(features)
        
        # Log prediction
        mlflow.log_metric("prediction_confidence", float(np.max(prediction[0])))

if __name__ == "__main__":
    main()