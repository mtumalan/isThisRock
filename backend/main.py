from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, Any

app = FastAPI()

class Features(BaseModel):
    features: Dict[str, Any]

@app.post("/predict/")
async def get_features(features: Features):
    print(f"Received features: {features.features}")
    # You can return a message or process the features further
    return {"message": "Features received successfully"}