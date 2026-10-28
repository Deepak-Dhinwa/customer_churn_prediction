import sys
from pathlib import Path

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from predict import load_model, predict_churn

app = FastAPI()
model = load_model()


class CustomerData(BaseModel):
    data: list


@app.get("/")
def read_root():
    return {"message": "Customer Churn Prediction API is running"}


@app.post("/predict")
def predict(data: CustomerData):
    input_df = pd.DataFrame(data.data)
    result_df = predict_churn(model, input_df)
    return result_df[["Churn_Pred", "Churn_Prob"]].to_dict(orient="records")
