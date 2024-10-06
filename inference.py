import xgboost as xgb
import pandas as pd
from main import create_features
from main import FEATURES

# Load the saved model
reg = xgb.XGBRegressor()
reg.load_model("C:\\Users\\Cesar\\Documents\\SpaceApps 2024\\model\\model")

# Assuming df is the new data
df_new = pd.read_csv("new_data.csv")
df_new = df_new.set_index("CALENDAR_DATE")
df_new.index = pd.to_datetime(df_new.index)

# Create the same features used during training
df_new = create_features(df_new)

# Use the same features that were used during training
X_new = df_new[FEATURES]

# Make predictions
predictions = reg.predict(X_new)

# Add predictions to the dataframe for comparison
df_new['prediction'] = predictions

df_new.to_csv("predictions.csv", columns=["prediction"])
