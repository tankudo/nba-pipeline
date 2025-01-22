from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from data.feature_store import FeatureStore
from models.model import NextBestActionModel
from utils.preprocessing import Preprocessor
import pandas as pd

app = FastAPI()

# Initialize components with config
CONFIG_PATH = "config/config.yaml"
feature_store = FeatureStore(CONFIG_PATH)
model = NextBestActionModel(CONFIG_PATH)
preprocessor = Preprocessor()

class PredictionRequest(BaseModel):
    user_id: str
    features: dict

class PredictionResponse(BaseModel):
    action: str
    probability: float

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        # Get user features from feature store
        user_features = feature_store.get_user_features(request.user_id)
        if not user_features:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Combine historical and current features
        features = {**user_features, **request.features}
        
        # Preprocess features
        df = pd.DataFrame([features])
        processed_features = preprocessor.preprocess_features(df)
        
        # Make prediction
        prediction = model.predict(processed_features)
        
        return PredictionResponse(
            action=str(np.argmax(prediction[0])),
            probability=float(np.max(prediction[0]))
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)