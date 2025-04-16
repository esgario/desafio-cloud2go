
import cloudpickle
import pandas as pd

from typing import Dict, Union, List
from fastapi import FastAPI
from pydantic import BaseModel


app = FastAPI()

# Load model
with open("./model/model.pkl", "rb") as fp:
    model = cloudpickle.load(fp)


class FeaturesInput(BaseModel):
    features: Union[Dict, List[Dict]]


@app.post("/predict")
def predict(data: FeaturesInput):
    if isinstance(data.features, dict):
        input_dicts = [data.features]
    else:
        input_dicts = data.features
    
    df = pd.DataFrame(input_dicts)
    preds = model.predict(df)
    return {"predictions": [int(pred) for pred in preds]}
