import json
import pandas as pd
import xgboost as xgb


class FeatureMapping():
    def __init__(self, mapping_path: str):
        with open(mapping_path, "rb") as fp:
            self._mapping = json.load(fp)
            
    def transform(self, df: pd.DataFrame):
        df_copy = df.copy()
        for col in self._mapping:
            if col not in df_copy:
                continue
            df_copy[col] = df_copy[col].map(self._mapping[col]).fillna(-1).astype(int)

        return df_copy


class SalePriceModel():
    def __init__(self, features: list, target: str, mapping):
        self._model = None
        self._features = features
        self._target = target
        self._mapping = mapping

    def train(self, df, **kwargs):
        # Prepare data
        X, y = df.drop(self._target, axis=1), df[self._target]
        X = self._mapping.transform(X)

        # Fit model
        self._model = xgb.XGBRegressor(objective="reg:absoluteerror", **kwargs)
        self._model.fit(X[self._features], y)
    
    def predict(self, X_test):
        X_test = self._mapping.transform(X_test)
        return self._model.predict(X_test[self._features])
