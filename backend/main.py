from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, Any
from RN_2 import trainModel, predictModel
import numpy as np

app = FastAPI()

model, scaler, encoder = trainModel()

class Features(BaseModel):
    features: Dict[str, Any]

@app.post("/predict/")
async def get_features(features: Features):
    X_test = np.array(list(features.features.values()))
    print(X_test)
    y_pred = predictModel(model, X_test, scaler)
    # Decode the encoded labels
    y_pred = encoder.inverse_transform(y_pred)
    return {"prediction": y_pred.tolist()}