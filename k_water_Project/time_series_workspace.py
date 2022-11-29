#####################
## import libraries
#####################
import numpy as np
import pandas as pd
import math

import statsmodels.api as sm
import statsmodels.tsa.api as smt
import statsmodels.formula.api as smf
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt

import itertools
import warnings


#####################
## data import
#####################
filename = '../input/pune_1965_to_2002.csv'

rainfall_data_matrix = pd.read_csv(filename) # csv file import
rainfall_data_matrix.head() # head()

#####################
## Set 'Year' as index
#####################
rainfall_data_matrix.set_index('Year', inplace=True)
rainfall_data_matrix.head()

########################################
## Transpose data for easy visualiztion
########################################
rainfall_data_matrix = rainfall_data_matrix.transpose()
rainfall_data_matrix

## Genearete dates from 1965-01(January 1965) to 2002-12(December 2002)
dates = pd.date_range(start='1965-01', freq='MS', periods=len(rainfall_data_matrix.columns)*12) # 날짜 생성할 수 있는 함수
dates

np.NaN