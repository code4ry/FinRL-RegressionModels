import numpy as np
import pandas as pd
from sklearn import svm
from sklearn import model_selection
from statsmodels.tools.eval_measures import mse
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
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

plt.scatter(yTest, yPredTest)
plt.xlabel("Actual Target Values")
plt.ylabel("Predicted Target Values")
plt.title("Actual vs. Predicted (Testing Set)")
plt.show()
# train model on one dataset, and test on another
df2Name = input("What is the name of your second dataset?: ")
df2 = pd.read_csv("/content/" + df2Name)

df2 = df2.dropna()
df2 = df2.dropna(axis=1)
df2 = df2.dropna(how='all')

X2 = df2[['Open', 'High', 'Low']]
y2 = df2[['Adj Close']]

xTrain2, xTest2, yTrain2, yTest2 = train_test_split(X2, y2, test_size=0.3, random_state=42)

lk = svm.SVR()
lk.fit(xTrain, yTrain)

yPredTest2 = lk.predict(xTest2)
meanErrorTest = mean_absolute_error(yTest2, yPredTest2)
print(meanErrorTest)

yPredTrain2 = lk.predict(xTrain2)
meanErrorTrain = mean_absolute_error(yTrain2, yPredTrain2)
print(meanErrorTrain)

plt.scatter(yTest2, yPredTest2)
plt.xlabel("Actual Target Values")
plt.ylabel("Predicted Target Values")
plt.title("Actual vs. Predicted (Testing Set)")
plt.show()