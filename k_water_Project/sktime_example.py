import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

# Read
df = pd.read_csv('dataset/Pedestrian_Counting_System_-_Monthly__counts_per_hour_.csv', usecols=['Date_Time', 'Sensor_Name', 'Hourly_Counts'])

df.head()

# Convert date to datetime
df['Date_Time'] = pd.to_datetime(df['Date_Time'])

# Group all sensors
df_grouped = df.groupby(['Date_Time']).agg({'Hourly_Counts': 'sum'}).rename(columns={'Hourly_Counts': 'Count_Pedestrians'})

# Aggregate weekly
df_weekly = df_grouped.resample('W').sum()

# Filter from the start of 2010 to end of 2019
df_weekly = df_weekly['2010-01-01': '2019-12-31']

df_weekly.index

df_weekly

y = df_weekly.squeeze() # prepare the data as a pandas Series

from sktime.forecasting.model_selection import temporal_train_test_split

y_train, y_test = temporal_train_test_split(y, test_size=26) # Predict from 1st July 2019

from sktime.forecasting.base import ForecastingHorizon

fh = ForecastingHorizon(y_test.index, is_relative=False)

from sklearn.linear_model import LinearRegression
from sktime.forecasting.compose import make_reduction

regressor = LinearRegression()
forecaster = make_reduction(regressor, window_length=52, strategy="recursive")

forecaster.fit(y_train)

y_pred = forecaster.predict(fh)

from xgboost import XGBRegressor

regressor = XGBRegressor(objective='reg:squarederror', random_state=42)
forecaster = make_reduction(regressor, window_length=52, strategy="recursive")

# Create an exogenous dataframe indicating the month
X = pd.DataFrame({'month': y.index.month}, index=y.index)
X = pd.get_dummies(X.astype(str), drop_first=True)
X_train, X_test = temporal_train_test_split(X, test_size=26) # Predict from 1st July 2019

# Fit
forecaster.fit(y=y_train, X=X_train)

y_train
X_train

# Predict
y_pred = forecaster.predict(fh=fh, X=X_test)

from sktime.forecasting.model_selection import SingleWindowSplitter

validation_size = 26
cv = SingleWindowSplitter(window_length=len(y)-validation_size, fh=validation_size)

from sktime.forecasting.model_selection import ForecastingRandomizedSearchCV
from sktime.forecasting.model_selection import ForecastingGridSearchCV

param_grid = {
    'estimator__max_depth': [3, 5, 6, 10, 15, 20],
    'estimator__learning_rate': [0.01, 0.1, 0.2, 0.3],
    'estimator__subsample': np.arange(0.5, 1.0, 0.1),
    'estimator__colsample_bytree': np.arange(0.4, 1.0, 0.1),
    'estimator__colsample_bylevel': np.arange(0.4, 1.0, 0.1),
    'estimator__n_estimators': [100, 500, 1000]
}

regressor = XGBRegressor(objective='reg:squarederror', random_state=42)
forecaster = make_reduction(regressor, window_length=52, strategy="recursive")

gscv = ForecastingRandomizedSearchCV(forecaster, cv=cv, param_distributions=param_grid, n_iter=100, random_state=42)


gscv.fit(y=y_train) # y_train만 있으면 작동함 / X_train이 같이 들어가면 오작동 / 근데 위에 fit은 잘 작동함  -> ?

gscv.get_tag


from sktime.forecasting.compose import TransformedTargetForecaster
from sktime.transformations.series.detrend import Deseasonalizer, Detrender
from sktime.forecasting.trend import PolynomialTrendForecaster

regressor = XGBRegressor(objective='reg:squarederror', random_state=42)

forecaster = TransformedTargetForecaster(
    [
        ("deseasonalize", Deseasonalizer(model="multiplicative", sp=52)),
        ("detrend", Detrender(forecaster=PolynomialTrendForecaster(degree=1))),
        ("forecast", make_reduction(regressor, window_length=52, strategy="recursive"),
        ),
    ]
)

param_grid = {
    'deseasonalize__model': ['multiplicative', 'additive'],
    'detrend__forecaster__degree': [1, 2, 3],
    'forecast__estimator__max_depth': [3, 5, 6, 10, 15, 20],
    'forecast__estimator__learning_rate': [0.01, 0.1, 0.2, 0.3],
    'forecast__estimator__subsample': np.arange(0.5, 1.0, 0.1),
    'forecast__estimator__colsample_bytree': np.arange(0.4, 1.0, 0.1),
    'forecast__estimator__colsample_bylevel': np.arange(0.4, 1.0, 0.1),
    'forecast__estimator__n_estimators': [100, 500, 1000]
}

gscv = ForecastingRandomizedSearchCV(forecaster, cv=cv, param_distributions=param_grid, n_iter=100, random_state=42)
# Fit
gscv.fit(y=y_train, X=X_train)

# Predict
y_pred = gscv.predict(fh=fh, X=X_test)



value = '1H'
aaa = pd.tseries.frequencies.to_offset(value)