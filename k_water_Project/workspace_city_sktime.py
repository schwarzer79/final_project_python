############################################################
## Import and Setting
############################################################
import copy

import numpy as np

np.random.seed(42)

import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

# some metrics and stats
from sklearn.metrics import mean_absolute_error as MAE
from scipy.stats import skew

# some utilities from the calendar package
from calendar import day_abbr, month_abbr, mdays

# fbprophet itself, we use here the version 0.3, release on the 3rd of June 2018
import prophet
prophet.__version__ # 1.1.1

# import some utility functions for data munging and plotting
import utils

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error

from prophet.diagnostics import cross_validation
from prophet.diagnostics import performance_metrics

from statsmodels.tsa.seasonal import seasonal_decompose

import xgboost

############################################################
## Import Data
############################################################

train_df = pd.read_csv('dataset/data_city/data_tr_city.csv')
test_df = pd.read_csv('dataset/data_city/data_ts_city.csv')
sample_df = pd.read_csv('dataset/data_city/sample_city.csv')

train_df.columns = ['ds', 'y']
test_df.columns = ['ds', 'y']

# 이상값 제거

train_df['ds'] = pd.to_datetime(train_df['ds'])
test_df['ds'] = pd.to_datetime(test_df['ds'])

train_df.info() # 8개 결측값

train_df.loc[train_df['y'] > 10000, 'y'] = np.NaN
train_df.loc[train_df['y'] <= 0,'y'] = np.NaN

## Anomaly detection with prophet
# seaborn을 사용하여 데이터 시각화 
sns.set(rc={'figure.figsize':(12,8)}) 
sns.lineplot(x=train_df['ds'], y=train_df['y']) 
plt.legend (['Amount'])
plt.show()

# Add seasonality
model = prophet.Prophet(changepoint_prior_scale = 0.5, interval_width=0.99, yearly_seasonality=True, weekly_seasonality=False, daily_seasonality = True)
# Fit the model on the training dataset
model.fit(train_df)

prediction = model.predict(train_df)
prediction.plot()
plt.show()

model.plot_components(prediction)
plt.show()

# 실제 값과 예측 값 병합 
performance = pd.merge(train_df, prediction[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], on='ds')

performance['anomaly'] = performance.apply(lambda rows: 1 if ((rows.y<rows.yhat_lower)|(rows.y>rows.yhat_upper)) else 0, axis = 1)
# 이상 징후 개수 확인 
performance['anomaly'].value_counts()

# 이상 현상 살펴보기 
anomalies = performance[performance['anomaly']==1].sort_values(by='ds') 
anomalies

# 이상값 대체 ---
for iter in anomalies.index :
    train_df.loc[train_df.index == iter,'y'] = np.NaN

##############################################################
## 계절성 확인
##############################################################

# help function

def missing_value_func(df, method, window = 5, halflife = 4) :
    df_copy = copy.deepcopy(df)
    if method == 'ffill' :
        df_copy['y'] = df_copy['y'].fillna(method = method)
    elif method == 'bfill' :
        df_copy['y'] = df_copy['y'].fillna(method = method)
    elif method == 'SMA' :
        df_copy['y'] = df_copy['y'].rolling(window=window, min_periods=1).mean()
    elif method == 'WMA' :
        df_copy['y'] = df_copy['y'].ewm(halflife=halflife).mean()
    elif method == 'linear' :
        df_copy['y'] = df_copy['y'].interpolate(option=method)
    elif method == 'spline' :
        df_copy['y'] = df_copy['y'].interpolate(option=method)
    else :
        df_copy['y'] = df_copy['y'].interpolate(option=method)
    #df_copy = df_copy.dropna().reset_index()
    return df_copy

train_df_Interpolation_time = missing_value_func(train_df, 'time')

train_df_Interpolation_time = train_df_Interpolation_time.set_index('ds')
train_df_Interpolation_time.index = pd.DatetimeIndex(train_df_Interpolation_time.index.values, freq=train_df_Interpolation_time.index.inferred_freq)

# resampling to Day + seasonal_decompose plotting
train_df_Interpolation_time = train_df_Interpolation_time['y'].resample('D').mean()
result = seasonal_decompose(train_df_Interpolation_time, model='additive')
result.plot()
plt.show()

# resampling to week + seasonal_decompose plotting
train_df_Interpolation_time = train_df_Interpolation_time['y'].resample('W').mean()
result = seasonal_decompose(train_df_Interpolation_time, model='additive')
result.plot()
plt.show()

####################################################################################
## Analysis with LinearRegression
####################################################################################

from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.forecasting.base import ForecastingHorizon
from sklearn.linear_model import LinearRegression

# train_df_Interpolation_time = pd.DataFrame(train_df_Interpolation_time)
test_dfc = test_df.set_index('ds')
test_dfc.index = pd.DatetimeIndex(test_dfc.index.values, freq=test_dfc.index.inferred_freq)

fh = ForecastingHorizon(test_dfc.index, is_relative=False)

regressor = LinearRegression()

from sktime.forecasting.compose import make_reduction
forecaster = make_reduction(regressor, window_length=60, strategy="recursive")

forecaster.fit(train_df_Interpolation_time)
y_pred = forecaster.predict(fh)

# plotting
from sktime.utils.plotting import plot_series
train_df_Interpolation_time = train_df_Interpolation_time.drop('index', axis=1)
y_pred = y_pred.drop('index',axis=1)
plot_series(train_df_Interpolation_time['2018-07-01':], test_dfc, y_pred, labels=["y_train", "y_test", "y_pred"], x_label='Date', y_label='Count pedestrians');
plt.show()

# metrics

from sktime.performance_metrics.forecasting import mean_absolute_percentage_error
print('MAE: %.4f' % mean_absolute_error(test_dfc, y_pred))

####################################################################################
## Analysis with XGBoost
####################################################################################

df = pd.concat([train_df_Interpolation_time, test_dfc], axis=0)
df = df.squeeze()

X = pd.DataFrame({'year': df.index.year, 'day' : df.index.day}, index=df.index)
X = pd.get_dummies(X.astype(str), drop_first=True)

X_train, X_test = temporal_train_test_split(X, test_size=8424)
y_train, y_test = temporal_train_test_split(df, test_size=8424)

# Fit
forecaster.fit(y=y_train, X=X_train)

# Predict
y_pred = forecaster.predict(fh=fh, X=X_test)

plot_series(y_train['2017-01-01':], y_test, y_pred, labels=["y_train", "y_test", "y_pred"], x_label='Date', y_label='Count pedestrians');
plt.show()

print('MAE: %.4f' % mean_absolute_error(y_test, y_pred)) # 158

####################################################################################
## Analysis with XGBoost + Tuning Hyperparameter
####################################################################################

from sktime.forecasting.model_selection import SingleWindowSplitter

validation_size = 7000
cv = SingleWindowSplitter(window_length=len(df)-validation_size, fh=validation_size)

from sktime.forecasting.model_selection import ForecastingRandomizedSearchCV

param_grid = {
    'estimator__max_depth': [3, 5, 6, 10, 15, 20],
    'estimator__learning_rate': [0.01, 0.1, 0.2, 0.3],
    'estimator__subsample': np.arange(0.5, 1.0, 0.1),
    'estimator__colsample_bytree': np.arange(0.4, 1.0, 0.1),
    'estimator__colsample_bylevel': np.arange(0.4, 1.0, 0.1),
    'estimator__n_estimators': [100, 500, 1000]
}

regressor = xgboost.XGBRegressor(objective='reg:squarederror', random_state=42)
forecaster = make_reduction(regressor, window_length=52, strategy="recursive")

gscv = ForecastingRandomizedSearchCV(forecaster, cv=cv, param_distributions=param_grid, n_iter=100, random_state=42)

# Fit
gscv.fit(y=y_train, X=X_train)

# Predict
y_pred = gscv.predict(fh=fh, X=X_test)