# -*- coding: utf-8 -*-
"""Ridge Regression Model.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1eQ4UZWcmITOHzIr6mp3yHH3f0owGRZOF#scrollTo=KIZ6SgnSaIp2
"""

#imports
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler

import numpy as np
import math
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

## parsing our stock market dataset
stock_datasetDJI = "^DJI.csv"

stock_dataDJI = pd.read_csv(stock_datasetDJI)

## removing our null values from the dataset
stock_dataDJI = stock_dataDJI.dropna()
stock_dataDJI = stock_dataDJI.dropna(axis=1)
stock_dataDJI = stock_dataDJI.dropna(how='all')

stock_dataDJI

## parsing stock market dataset
stock_datasetSANDP = "sandp500.csv"

stock_dataSANDP = pd.read_csv(stock_datasetSANDP)

#removing our null values from the dataset
stock_dataSANDP = stock_dataSANDP.dropna()
stock_dataSANDP = stock_dataSANDP.dropna(axis=1)
stock_dataSANDP = stock_dataSANDP.dropna(how='all')

stock_dataSANDP

#plotting Open vs Adj Close in DJI
plt.figure(figsize = (18,9))
plt.xlabel("Open")
plt.ylabel("Adjusted Close")
plt.scatter(stock_dataDJI["Open"], stock_dataDJI["Adj Close"])

#plotting Open vs Adj Close in S&P500
plt.figure(figsize = (18,9))
plt.xlabel("Open")
plt.ylabel("Adjusted Close")
plt.scatter(stock_dataSANDP["Open"], stock_dataSANDP["Adj Close"])

df = pd.read_csv("/content/sandp500.csv")
headers = df.head(0)
print(headers)
df = df.dropna()
df = df.dropna(axis=1)
df= df.dropna(how='all')
numRows = df.shape[0]

xDate = pd.to_datetime(df['Date'])  # Convert 'Date' column to datetime format
xOpen = df[['Open']]
xHigh = df[['High']]
xLow = df[['Low']]
xVol = df[['Volume']]
xClose = df[['Close']]

y = df[['Adj Close']]

#Visualizing open prices to adjusted close price
plt.scatter(xOpen, y)
plt.xlabel('Open Prices')
plt.ylabel('Adjusted Close Prices')
plt.title('Comparison 1')
plt.show()

#Visualizing highest prices to adjusted close price
plt.scatter(xHigh, y)
plt.xlabel('Highest Prices')
plt.ylabel('Adjusted Close Prices')
plt.title('Comparison 2')
plt.show()

#Visualizing lowest prices to adjusted close price
plt.scatter(xLow, y)
plt.xlabel('Lowest Prices')
plt.ylabel('Adjusted Close Prices')
plt.title('Comparison 3')
plt.show()

#Visualizing stock volume to adjusted close price
plt.scatter(xVol, y)
plt.xlabel('Stock Volume')
plt.ylabel('Adjusted Close Prices')
plt.title('Comparison 4')
plt.show()

#Visualizing open price per day
plt.scatter(xDate, xOpen)
plt.xlabel('Date')
plt.ylabel('Opening price')
plt.title('Comparison 5')
plt.show()

#Visualizing closing price per day
plt.scatter(xDate, xClose)
plt.xlabel('Date')
plt.ylabel('Closing price')
plt.title('Comparison 6')
plt.show()

"""
After visualizing our data set, we can see that volume stock does not provide a meaningful prediction on our adjusted close prices.
Our models visualizes the stock prices from 2022-01-03 to 2023-12-29, for both data sets.
"""

#Splitting for train test
X = df[['Open', 'High', 'Low']]
xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=0.3, random_state=42)

ridge = Ridge(alpha=0.1)

ridge.fit(xTrain, yTrain)

yPred = ridge.predict(xTest)
yPred_Train = ridge.predict(xTrain)

mse_test = mean_squared_error(yTest, yPred)
mse_train = mean_squared_error(yTrain, yPred_Train)

print('MSE Test: ', mse_test)
print('MSE Train: ', mse_train)