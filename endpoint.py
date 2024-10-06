import xgboost as xgb
import pandas as pd
import requests
import json
import os

FEATURES = ["quarter", "month", "year"]
TARGET = "EVI"

model_name = "model"
current_dir = os.path.abspath(os.getcwd())

model_path = f"{current_dir}/{model_name}"

# Load the saved model
reg = xgb.XGBRegressor()
reg.load_model(model_path)

km_above_below = 0
km_left_right = 0

url = "https://modis.ornl.gov/rst/api/v1/"
header = {
    "Accept": "application/json"
}  # Use following for a csv response: header = {'Accept': 'text/csv'}

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

def merge(df1, df2, attr='CALENDAR_DATE'):
    # Merge the two datasets on 'CALENDAR_DATE', using an outer join to retain all dates
    merged_df = pd.merge(df1, df2, on=attr, how='outer')

    # Sort by 'CALENDAR_DATE' to ensure proper alignment
    merged_df.sort_values('CALENDAR_DATE', inplace=True)

    # Forward fill the missing values
    merged_df.fillna(method='ffill', inplace=True)
    return merged_df


def get_preds(longitude=-115, latitude=32.6428056):
    prod = "MOD11A2"
    band = "Clear_sky_days"

    dates_url = url + f"{prod}/dates?longitude={longitude}&latitude={latitude}"
    response = requests.get(dates_url)
    dates = json.loads(response.text)["dates"]

    dates = dates[-6:]

    modis_dates = [i["modis_date"] for i in dates]
    calendar_dates = [i["calendar_date"] for i in dates]

    date_equivalence = {
        modis_dates[i]: calendar_dates[i] for i in range(len(modis_dates))
    }

    clear_sky_list = []
    data_url = (
        url
        + f"{prod}/subset?latitude={latitude}&longitude={longitude}&band={band}&startDate={modis_dates[0]}&endDate={modis_dates[-1]}&kmAboveBelow={km_above_below}&kmLeftRight={km_left_right}"
    )
    response = requests.get(data_url)

    clear_sky_subset = json.loads(response.text)["subset"]
    for subset in clear_sky_subset:
        clear_sky_list.append({
            "CALENDAR_DATE": date_equivalence.get(subset["modis_date"]),
            "LST_DAY": subset["data"][0]
        })


    band = "Clear_sky_nights"
    dark_sky_list = []
    data_url = (
        url
        + f"{prod}/subset?latitude={latitude}&longitude={longitude}&band={band}&startDate={modis_dates[0]}&endDate={modis_dates[-1]}&kmAboveBelow={km_above_below}&kmLeftRight={km_left_right}"
    )
    response = requests.get(data_url)

    dark_sky_subset = json.loads(response.text)["subset"]
    for subset in dark_sky_subset:
        dark_sky_list.append({
            "CALENDAR_DATE": date_equivalence.get(subset["modis_date"]),
            "LST_NIGHT": subset["data"][0]
        })
    
    prod = "MOD13Q1"
    band = "250m_16_days_EVI"

    dates_url = url + f"{prod}/dates?longitude={longitude}&latitude={latitude}"
    response = requests.get(dates_url)
    mod13q1_dates = json.loads(response.text)["dates"]

    filtered_mod13q1_dates = [  
        date for date in mod13q1_dates if date["modis_date"] >= modis_dates[0]
    ]

    modis_dates_16 = [i["modis_date"] for i in filtered_mod13q1_dates]
    #calendar_dates_16 = [i["calendar_date"] for i in filtered_mod13q1_dates]

    evi_list = []
    data_url = (
        url
        + f"{prod}/subset?latitude={latitude}&longitude={longitude}&band={band}&startDate={modis_dates_16[0]}&endDate={modis_dates_16[-1]}&kmAboveBelow={km_above_below}&kmLeftRight={km_left_right}"
    )
    response = requests.get(data_url)

    evi_subset = json.loads(response.text)["subset"]
    for subset in evi_subset:
        evi_list.append({
            "CALENDAR_DATE": date_equivalence.get(subset["modis_date"]),
            "EVI": subset["data"][0]
        })

    band="250m_16_days_NDVI"
    ndvi_list = []
    data_url = (
        url
        + f"{prod}/subset?latitude={latitude}&longitude={longitude}&band={band}&startDate={modis_dates[0]}&endDate={modis_dates[-1]}&kmAboveBelow={km_above_below}&kmLeftRight={km_left_right}"
    )
    response = requests.get(data_url)

    ndvi_subset = json.loads(response.text)["subset"]
    for subset in ndvi_subset:
        ndvi_list.append({
            "CALENDAR_DATE": date_equivalence.get(subset["modis_date"]),
            "NDVI": subset["data"][0]
        })

    # Convert LST data to a DataFrame
    lst_day_df = pd.DataFrame(clear_sky_list)
    lst_night_df = pd.DataFrame(dark_sky_list)

    # Merge LST Day and Night DataFrames
    lst_df = pd.merge(lst_day_df, lst_night_df, on="CALENDAR_DATE", how="outer")

    evi_df = pd.DataFrame(evi_list)
    ndvi_df = pd.DataFrame(ndvi_list)

    evi_ndvi_df = pd.merge(evi_df, ndvi_df, on="CALENDAR_DATE", how="outer")
    
    data = merge(lst_df, evi_ndvi_df)

    data = data.set_index("CALENDAR_DATE")
    data.index = pd.to_datetime(data.index)

    data = create_features(data)

    X_new = data[FEATURES]

    predictions = reg.predict(X_new)
    data["prediction"] = predictions

    #result_df = data[["prediction"]].reset_index()

    return data

get_preds()