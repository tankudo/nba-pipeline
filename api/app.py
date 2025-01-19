from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np

app = FastAPI()

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
            raise HTTPException(status_code=404, message="User not found")
        
        # Make prediction
        features = {**user_features, **request.features}
        prediction = model.predict(np.array([list(features.values())]))
        
        return PredictionResponse(
            action=str(np.argmax(prediction[0])),
            probability=float(np.max(prediction[0]))
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))