import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import prophet
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error

## Data Import ##
train_df = pd.read_csv('dataset/data_apo/data_tr_apo.csv')
test_df = pd.read_csv('dataset/data_apo/data_ts_apo.csv')
combine = [train_df, test_df]

train_df['datetime'] = pd.to_datetime(train_df['datetime'])

## Distribution ##
train_df.columns.values

train_df[train_df['구미 아포배수지 유출유량 적산차'] > 3069]

copy = train_df
copy[copy['구미 아포배수지 유출유량 적산차'] > 1000] = 1000
copy[copy['구미 아포배수지 유출유량 적산차'] > 1000]

copy[copy['구미 아포배수지 유출유량 적산차'] < 0] = 0

train_df.describe()

train_df.info()

fig = px.line(train_df, x = 'datetime', y = '구미 아포배수지 유출유량 적산차')
fig.show()

sns.lineplot(data=copy)
plt.show()

##############################################################################
## anomaly detection with prophet
##############################################################################
train_df.columns = ['ds', 'y']
train_df['ds'] = pd.to_datetime(train_df['ds'])

train_df.info() # 8개 결측값

# seaborn을 사용하여 데이터 시각화 
sns.set(rc={'figure.figsize':(12,8)}) 
sns.lineplot(x=train_df['ds'], y=train_df['y']) 
plt.legend (['Amount'])
plt.show()


## Anomaly detection with prophet
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

# 이상치로 판정된 값을 np.NaN으로 변경
for iter in anomalies.index :
    train_df.loc[train_df.index == iter,'y'] = np.NaN

for iter in anomalies.index :
    print(train_df[train_df.index == iter])

