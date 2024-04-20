import numpy as np
import pandas as pd
from sklearn import svm
from sklearn import model_selection
from statsmodels.tools.eval_measures import mse
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

df1Name = input("What is the name of your first dataset?: ")
df = pd.read_csv("/content/" + df1Name)

df = df.dropna()
df = df.dropna(axis=1)
df= df.dropna(how='all')

numeric_columns = df.select_dtypes(include=[np.number])

#scaling the data
df[numeric_columns.columns] = (numeric_columns - numeric_columns.min()) / (numeric_columns.max() - numeric_columns.min())
df.head()

X = df[['Open', 'High', 'Low']]
y = df[['Adj Close']]
xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=0.3, random_state=42)

yTrain = yTrain.values.ravel()
yTest = yTest.values.ravel()

#linear kernel
lk = svm.SVR()
lk.fit(xTrain, yTrain)

yPredTrain = lk.predict(xTrain)
yPredTest = lk.predict(xTest)

meanErrorTest = mean_absolute_error(yTest, yPredTest)
meanErrorTrain = mean_absolute_error(yTrain, yPredTrain)

print(meanErrorTest)
print(meanErrorTrain)
