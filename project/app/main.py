from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from predict import load_model, predict_churn

app = FastAPI()

# Load trained model at startup
model = load_model("models/lgb_model.pkl")

class CustomerData(BaseModel):
    data: list  # List of dicts for batch prediction

@app.get("/")
def read_root():
    return {"message": "Customer Churn Prediction API is running"}

@app.post("/predict")
def predict(data: CustomerData):
    """
    Predict churn from JSON input data
    """
    # Convert input JSON to DataFrame
    input_df = pd.DataFrame(data.data)

    # Use predict.py function to get predictions
    result_df = predict_churn(model, input_df)

    # Return predictions and probabilities as JSON
    return result_df[['Churn_Pred', 'Churn_Prob']].to_dict(orient='records')
