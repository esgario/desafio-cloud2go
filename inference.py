import requests
import pandas as pd
import numpy as np

df_test = pd.read_csv("data/test.csv")

data = df_test.replace({np.nan: None})
data = data.replace({np.nan: None}).to_dict(orient="records")
data = {"features": data}

response = requests.post("http://127.0.0.1:8000/predict", json=data)

df_test["SalePrice"] = response.json()["predictions"]
df_test.to_csv("data/predictions.csv")

print(response.json())