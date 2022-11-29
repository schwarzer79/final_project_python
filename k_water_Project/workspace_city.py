import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import prophet
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from matplotlib import rcParams
import copy
import matplotlib.dates as mdates

#####################################################################################################
## Prophet을 이용한 일변량 시계열 분석
#####################################################################################################

## Data Import ##
train_df = pd.read_csv('dataset/data_city/data_tr_city.csv')
test_df = pd.read_csv('dataset/data_city/data_ts_city.csv')
combine = [train_df, test_df]

## Data Distribution ##
train_df.columns.values

train_df.head()

train_df.columns = ['ds', 'y']
train_df['ds'] = pd.to_datetime(train_df['ds'])

train_df.info() # 8개 결측값

## Anomaly detection with prophet
# seaborn을 사용하여 데이터 시각화 
sns.set(rc={'figure.figsize':(12,8)}) 
sns.lineplot(x=train_df['ds'], y=train_df['y']) 
plt.legend (['Amount'])
plt.show()

# Add seasonality
model = prophet.Prophet(interval_width=0.99, yearly_seasonality=True, weekly_seasonality=True)
# Fit the model on the training dataset
model.fit(train_df)

prediction = model.predict(train_df)
prediction.plot()
plt.show()

model.plot_components(prediction)
plt.show()

# 실제 값과 예측 값 병합 
performance = pd.merge(train_df, prediction[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], on='ds')
# MAE 값 확인 
#performance_MAE = mean_absolute_error(performance['y'], performance['yhat']) 
#print(f'The MAE for the model is {performance_MAE}')
# MAPE 값 확인 
#performance_MAPE = mean_absolute_percentage_error(performance['y'], performance['yhat']) 
#print(f'The MAPE for the model is {performance_MAPE}')

performance['anomaly'] = performance.apply(lambda rows: 1 if ((rows.y<rows.yhat_lower)|(rows.y>rows.yhat_upper)) else 0, axis = 1)
# 이상 징후 개수 확인 
performance['anomaly'].value_counts()

# 이상 현상 살펴보기 
anomalies = performance[performance['anomaly']==1].sort_values(by='ds') 
anomalies

# 이상 징후 시각화 
sns.scatterplot(x='ds', y='y', data=performance, hue='anomaly') 
sns.lineplot(x='ds', y='yhat', data=performance, color='black')
plt.show()

######################################
## Replace anomalies
######################################
for iter in anomalies.index :
    train_df.loc[train_df.index == iter,'y'] = np.NaN

######################################
## Anomalies detection 2
######################################

# 이상 징후 시각화 
sns.scatterplot(x='ds', y='y', data=performance2, hue='anomaly') 
sns.lineplot(x='ds', y='yhat', data=performance2, color='black')
plt.show()


train_df[train_df['y'] > 2000]
"""
                      ds       y
3513 2017-05-27 10:00:00  4999.0 ## 오전 10시에 이 정도의 물 사용량은 이상치일 확률 높음 / 별다른 사건사고 없음
3533 2017-05-28 06:00:00  3357.0 ## 마찬가지로 새벽 6시
3542 2017-05-28 15:00:00  2550.0 ## 오후 3시.. 이거는 추세를 봐야할 수 있음
3554 2017-05-29 03:00:00  2334.0 
3607 2017-05-31 08:00:00  2591.0

 -> 각 시간 별 물 사용량의 평균값을 비교해서 현실적으로 맞는 값인지 확인할 필요성이 있어보임
"""

train_df[train_df['y'] > 1000]
"""
                       ds       y
3468  2017-05-25 13:00:00  1214.0
3485  2017-05-26 06:00:00  1105.0
3491  2017-05-26 12:00:00  1295.0
3513  2017-05-27 10:00:00  4999.0
3533  2017-05-28 06:00:00  3357.0
3542  2017-05-28 15:00:00  2550.0
3554  2017-05-29 03:00:00  2334.0
3593  2017-05-30 18:00:00  1679.0
3607  2017-05-31 08:00:00  2591.0
10208 2018-03-02 09:00:00  1077.0
12519 2018-06-06 16:00:00  1209.0
26078 2019-12-23 15:00:00  1274.0
26079 2019-12-23 16:00:00  1154.0
"""

# anomaly check
# 2017-05-25
train_df.loc[(train_df['ds'].dt.year == 2017) & (train_df['ds'].dt.month == 5) & (train_df['ds'].dt.day == 25)]
"""
                      ds       y
3455 2017-05-25 00:00:00     0.0
3456 2017-05-25 01:00:00   567.0
3457 2017-05-25 02:00:00     0.0
3458 2017-05-25 03:00:00   247.0
3459 2017-05-25 04:00:00     0.0
3460 2017-05-25 05:00:00   134.0
3461 2017-05-25 06:00:00     0.0
3462 2017-05-25 07:00:00    88.0
3463 2017-05-25 08:00:00     0.0
3464 2017-05-25 09:00:00   595.0
3465 2017-05-25 10:00:00   180.0
3466 2017-05-25 11:00:00     0.0
3467 2017-05-25 12:00:00     0.0
3468 2017-05-25 13:00:00  1214.0
3469 2017-05-25 14:00:00     0.0
3470 2017-05-25 15:00:00   410.0
3471 2017-05-25 16:00:00     0.0
3472 2017-05-25 17:00:00   707.0
3473 2017-05-25 18:00:00     0.0
3474 2017-05-25 19:00:00     0.0
3475 2017-05-25 20:00:00   536.0
3476 2017-05-25 21:00:00     0.0
3477 2017-05-25 22:00:00   440.0
3478 2017-05-25 23:00:00   275.0
"""

# 2017-05-26
train_df.loc[(train_df['ds'].dt.year == 2017) & (train_df['ds'].dt.month == 5) & (train_df['ds'].dt.day == 26)]
"""
                    ds       y
3479 2017-05-26 00:00:00   398.0
3480 2017-05-26 01:00:00     0.0
3481 2017-05-26 02:00:00     0.0
3482 2017-05-26 03:00:00     0.0
3483 2017-05-26 04:00:00     0.0
3484 2017-05-26 05:00:00     0.0
3485 2017-05-26 06:00:00  1105.0
3486 2017-05-26 07:00:00   139.0
3487 2017-05-26 08:00:00     0.0
3488 2017-05-26 09:00:00     0.0
3489 2017-05-26 10:00:00     0.0
3490 2017-05-26 11:00:00     0.0
3491 2017-05-26 12:00:00  1295.0
3492 2017-05-26 13:00:00     0.0
3493 2017-05-26 14:00:00     0.0
3494 2017-05-26 15:00:00     0.0
3495 2017-05-26 16:00:00     0.0
3496 2017-05-26 17:00:00     0.0
3497 2017-05-26 18:00:00     0.0
3498 2017-05-26 19:00:00     0.0
3499 2017-05-26 20:00:00     0.0
3500 2017-05-26 21:00:00     0.0
3501 2017-05-26 22:00:00     0.0
3502 2017-05-26 23:00:00     0.0
"""





# y > 1000인 case NaN으로 변경
train_df.loc[train_df['y'] > 1000, 'y'] = np.NaN

# y = 0
train_df.loc[train_df['y'] == 0, 'y'].index

#############################################
## Visualization
#############################################

## 각 시간에 대한 column 생성
train_df['year'] = train_df['ds'].dt.year
train_df['month'] = train_df['ds'].dt.month
train_df['day'] = train_df['ds'].dt.day
train_df['hour'] = train_df['ds'].dt.hour
train_df['dayofweek'] = train_df['ds'].dt.dayofweek # 월 = 0

train_df = train_df.set_index('ds')

for iter in range(7) :
    if iter < 5 :
        train_df.loc[train_df['dayofweek'] == iter, 'weekend'] = 'Weekday'
    else :
        train_df.loc[train_df['dayofweek'] == iter, 'weekend'] = 'Weekend'

## Helf function
def water_use_line(args, nbins) :
    sns.lineplot(data=train_df, x=args, y='y')
    plt.ylabel('Water Usage')
    plt.locator_params(nbins=nbins)
    plt.title(f"{args} Water Usage")
    plt.show()

def water_use_bar(args) :
    sns.barplot(data=train_df, x=args, y='y')
    plt.ylabel('Water Usage')
    plt.title(f"{args} Water Usage")
    plt.show()

## 시간별 물 사용량 시각화

water_use_line('hour', 24) # 40 ~ 380 사이의 값을 가지며 가장 낮은 시간은 4~6시 사이 가장 높은 시간은 20~22시 사이 상승하다가 22시에 max

## 연도별 물 사용량 visualization

water_use_line('year',6) # 전반적으로 상승하는 trend를 가지고 있으며 범위가 210~260 사이를 유지

## 월별 물 사용량 visualization

water_use_line('month', 13) # 6~8월달의 물 사용량이 가장 많으며 겨울철(12~2월)이 낮은 모양

## 주말/평일 물 사용량 visualization

water_use_bar('weekend') # 주말 사용량 < 평일 사용량

train_df.plot(subplots=True, figsize = (10,12))
plt.show()

####################################################
## 결측값 채우기
####################################################

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
    df_copy = df_copy.dropna().reset_index()
    return df_copy

# ffill()을 이용한 채우기

train_df_ffill = missing_value_func(train_df, 'ffill')

# bfill()

train_df_bfill = missing_value_func(train_df, 'bfill')

# Simple Moving Average

train_df_SMA = missing_value_func(train_df, 'SMA')

# Exponential Weighted Moving Average

train_df_EWMA = missing_value_func(train_df, 'EWMA')

# Interpolation - linear

train_df_Interpolation_linear = missing_value_func(train_df, 'linear')

# Interpolation - Spline

train_df_Interpolation_spline = missing_value_func(train_df,'spline')

# Interpolation - time

train_df_Interpolation_time = missing_value_func(train_df, 'time')

####################################################
## EDA
####################################################

# subplot을 이용한 계절성 탐색 -> 범위가 넓어 효과 x
train_df_ffill['y'].plot(subplots=True, figsize=(10,12))
plt.show()

# resampling 후 barplot
train_df_ffill_day = train_df_ffill.resample('D').mean()
fig, ax = plt.subplots(figsize=(10,6))
#ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%D'))
sns.lineplot(train_df_ffill_day.index, train_df_ffill_day['y'])
plt.tick_params(axis='x',labelsize=15,rotation=90)
plt.tight_layout()
plt.show()

# percent change

train_df_ffill_day.loc[:, 'pct_change'] = train_df_ffill['y'].pct_change()*100
fig, ax = plt.subplots()
train_df_ffill_day['pct_change'].plot(kind='bar', color='black', ax=ax)
#ax.xaxis.set_major_locator(mdates.WeekdayLocator())
plt.xticks(rotation=45)
ax.legend()
plt.show()

####################################################
## 시계열 분해
####################################################

import statsmodels.api as sm
from matplotlib import rcParams

decomposition = sm.tsa.seasonal_decompose(train_df_ffill, model = 'additive')
decomposition.plot()
plt.show()
