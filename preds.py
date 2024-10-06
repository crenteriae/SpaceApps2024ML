import pandas as pd
import numpy as np
from datetime import date
from sklearn.preprocessing import MinMaxScaler
import os
import xgboost as xgb

scaler = MinMaxScaler()

model_name = "model"
current_dir = os.path.abspath(os.getcwd())

model_path = f"{current_dir}/{model_name}"
test_data_path = f"{current_dir}/test_data.csv"

# Load the saved model
reg = xgb.XGBRegressor()
reg.load_model(model_path)

def create_features(df):
    """
    Create time series features based on time series index.
    """
    df = df.copy()
    #     df['hour'] = df.index.hour
    #     df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df["month"] = df.index.month
    df["year"] = df.index.year
    df["dayofyear"] = df.index.dayofyear
    df["dayofmonth"] = df.index.day
    df["weekofyear"] = df.index.isocalendar().week
    return df

def get_preds():
    df = pd.read_csv(test_data_path)
    df = df.set_index("CALENDAR_DATE")
    df.index = pd.to_datetime(df.index)

    df = create_features(df)
    FEATURES = ["quarter", "month", "year"]
    df = df[FEATURES]
    scaler.fit_transform(df)

    future_dates = pd.date_range(start=date.today(), periods=24, freq='SME')

    future_df = pd.DataFrame(index=future_dates)
    future_df = create_features(future_df)

    future_X = future_df[["quarter", "month", "year"]]
    future_X_scaled = scaler.transform(future_X)

    future_df['prediction'] = reg.predict(future_X_scaled)

    return future_df