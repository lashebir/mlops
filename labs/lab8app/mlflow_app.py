from fastapi import FastAPI
import uvicorn
import mlflow, mlflow.pyfunc
import joblib
from pydantic import BaseModel
import pandas as pd
from sklearn.datasets import load_wine

app = FastAPI(
    title="MLFlow Model by Leah",
    description="Predictions based on selected model from MLFlow",
    version="0.1",
)

class request_body(BaseModel):
    model : str

# Defining path operation for root endpoint
@app.get('/')
def main():
	return {'message': 'Provide the model and features to get predictions'}

# Defining path operation for /predict endpoint
@app.post('/predict')
def load_n_predict(data : request_body):
    model_pipeline = mlflow.pyfunc.load_model(data.model)

    wine = load_wine()
    X = pd.DataFrame(data = wine.data, columns=wine.feature_names)
    predictions = model_pipeline.predict(X)
    return {'Predictions': predictions.tolist()}
