""" code for your API here. """

import pickle
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from starter.ml.data import process_data

# Config
config = {
    'model_path': './model_enc_lb.pkl',
    'data_path': './data/census.csv',
    'cat_features': [
        "workclass",
        "education",
        "marital_status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native_country",
    ]
}

# Instatiate the app
app = FastAPI()

# Load model and encoder
with open(config['model_path'], 'rb') as f:
    model, encoder, lb = pickle.load(f)

# Declare BaseModel
class InputData(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str

    class Config:
        schema_extra = {
            'example': {
                'age': 32,
                'workclass': 'State-gov',
                'fnlgt': 141297,
                'education': 'Bachelors',
                'education_num': 9,
                'marital_status': 'Married-civ-spouse',
                'occupation': 'Exec-managerial',
                'relationship': 'Wife',
                'race': 'White',
                'sex': 'Female',
                'capital_gain': 10000,
                'capital_loss': 5000,
                'hours_per_week': 35,
                'native_country': 'China'
            }
        }


# Define a GET on the specified endpoint
@app.get("/")
async def say_hello():
    print('holu')
    return {'greeting': 'hello world'}


@app.post('/inference')
async def run_inference(input_data: InputData):
    print(input_data.dict())
    df = pd.DataFrame.from_dict({k: [v] for k, v in input_data.dict().items()})
    X, _, _ , _ = process_data(
        df,
        config['cat_features'],
        label=None,
        encoder=encoder,
        training=False,
        lb=lb
    )
    pred = model.predict(X)

    return '<=50K' if pred == 0 else '>50K'
