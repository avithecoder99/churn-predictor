from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# Load model
model = joblib.load("app/model/churn_model.pkl")

# Define input schema
class ChurnInput(BaseModel):
    age: int
    balance: float

# Define route
@app.post("/predict")
def predict_churn(data: ChurnInput):
    features = np.array([[data.age, data.balance]])
    prediction = model.predict(features)
    return {"churn": int(prediction[0])}
