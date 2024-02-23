# -*- coding: utf-8 -*-
"""polynomialRegressionRCOS.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1KfDLB9ESqn62hD-HCSLPTIh5atl2On50
"""

#imports
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

import numpy as np
import math

data = pd.read_csv("/content/sandp500.csv")
headers = data.head(0)
print(headers)

xDate = data[['Date']]
xOpen = data[['Open']]
xHigh = data[['High']]
xLow = data[['Low']]
xVol = data[['Volume']]

y = data[['Adj Close']]

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
plt.title('Comparison 3')
plt.show()

"""After visualizing our data set, we can see that volume stock does not provide a meaningful prediction on our adjusted close prices.
Our models visualizes the stock prices from 2022-01-03 to 2023-12-29.

"""