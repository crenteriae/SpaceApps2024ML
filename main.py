import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import xgboost as xgb

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

color_pal = sns.color_palette()
plt.style.use("fivethirtyeight")

df = pd.read_csv("C:\\Users\\Cesar\\Documents\\SpaceApps 2024\\merged_data.csv")
df = df.set_index("CALENDAR_DATE")
df.index = pd.to_datetime(df.index)

df.plot(style='.',
        figsize=(15, 5),
        color=color_pal[0],
        title='INDX')
plt.show()

train = df.loc[df.index < "01-01-2022"]
test = df.loc[df.index >= "01-01-2022"]

fig, ax = plt.subplots(figsize=(15, 5))
train.plot(ax=ax, label='Training Set', title='Data Train/Test Split')
test.plot(ax=ax, label='Test Set')
ax.axvline('01-01-2022', color='black', ls='--')
ax.legend(['Training Set', 'Test Set'])
plt.show()


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


df = create_features(df)

fig, ax = plt.subplots(figsize=(10, 8))
sns.boxplot(data=df, x='month', y='EVI')
ax.set_title('EVI by Month')
plt.show()

train = create_features(train)
test = create_features(test)

FEATURES = ["quarter", "month", "year"]
TARGET = "EVI"

X_train = train[FEATURES]
Y_train = train[TARGET]

X_test = test[FEATURES]
Y_test = test[TARGET]

# Normalize the feature sets
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# param_grid = {
#     'n_estimators': [500, 1000, 1500],
#     'max_depth': [3, 5, 7],
#     'learning_rate': [0.01, 0.05, 0.1],
#     'subsample': [0.6, 0.8, 1.0],
#     'colsample_bytree': [0.6, 0.8, 1.0],
#     'min_child_weight': [1, 3, 5]
# }

# xgb_reg = xgb.XGBRegressor(objective='reg:squarederror')
# random_search = RandomizedSearchCV(estimator=xgb_reg, param_distributions=param_grid, n_iter=50, scoring='neg_mean_squared_error', cv=5, verbose=1)
# random_search.fit(X_train, Y_train)
# print(f"Best params: {random_search.best_params_}")
# exit()


reg = xgb.XGBRegressor(
    min_child_weight=5,
    colsample_bytree=1.0,
    base_score=0.5,
    booster="gbtree",
    n_estimators=1000,
    early_stopping_rounds=50,
    objective="reg:squarederror",
    max_depth=7,
    max_leaves=0,
    learning_rate=0.1,
    subsample=0.8
)
reg.fit(X_train, Y_train, eval_set=[(X_train, Y_train), (X_test, Y_test)], verbose=100)

test['prediction'] = reg.predict(X_test)
df = df.merge(test[['prediction']], how='left', left_index=True, right_index=True)
ax = df[['EVI']].plot(figsize=(15, 5))
df['prediction'].plot(ax=ax, style='.')
plt.legend(['Truth Data', 'Predictions'])
ax.set_title('Raw Dat and Prediction')
plt.show()

score = np.sqrt(mean_squared_error(test['EVI'], test['prediction']))
print(f'RMSE Score on Test set: {score:0.2f}')

reg.save_model("model")
