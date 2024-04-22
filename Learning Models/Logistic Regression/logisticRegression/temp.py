import pandas as pd
import numpy as num
import matplotlib.pyplot as plt
import csv
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error


## parsing our stock market dataset
stock_dataset = "Learning Models\Logistic Regression\logisticRegression\^DJI.csv"
sp_dataset = "Learning Models\Lasso Regression\sandp500.csv"
stock_data = pd.read_csv(stock_dataset)
sp_500 = pd.read_csv(sp_dataset)

## removing our null values from the dataset
stock_data = stock_data.dropna()
stock_data = stock_data.dropna(axis=1)
stock_data = stock_data.dropna(how='all')

sp_500 = sp_500.dropna()
sp_500 = sp_500.dropna(axis=1)
sp_500 = sp_500.dropna(how='all')

stock_data["Tomorrow"] = stock_data["Close"].shift(-1)
sp_500["Tomorrow"] = sp_500["Close"].shift(-1)
sp_500["Target"] = (sp_500["Tomorrow"] > sp_500["Close"]).astype(int)

# Create a DataFrame with the original time series and the binary target variable
df = pd.DataFrame({"Close": stock_data["Close"], "Target": stock_data["Target"]})
sp_df = pd.DataFrame({"Close": sp_500["Close"], "Target": sp_500["Target"]})

# Feature Engineering: Adding a 7-day moving average
df['MA7'] = df['Close'].rolling(window=7).mean()
sp_df['MA7'] = sp_df['Close'].rolling(window=7).mean()

# Handling Missing Data: Dropping rows with NaN values
df.dropna(inplace=True)
sp_df.dropna(inplace=True)

# Splitting the data into features (X) and target variable (y)
x = df[['MA7']]
y = df['Target']

sp_X = sp_df[['MA7']]
sp_Y = sp_df['Target']

# Splitting the data into training and testing sets
train_size = int(len(df) * 0.8)  # 80% for training
X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=False)
spX_train, spX_test, spY_train, spY_test = train_test_split(sp_X, sp_Y, test_size=0.2, shuffle=False)


# Creating and training a logistic regression model
model = LogisticRegression(C=0.01) # adjusting C-value to 0.01 to combat overfitting and add regularization
cv_scores = cross_val_score(model, sp_X, sp_Y, scoring='accuracy', cv=5)  # 5-fold cross-validation

# Training the initial model with training data from DJI
model.fit(X_train, y_train)

# Making predictions on the train and test set
spY_train_pred = model.predict(spX_train)
spY_test_pred = model.predict(spX_test)

# Evaluating mean squared error
mse = mean_squared_error(spY_test, spY_test_pred)
print ("Mean Squared Error before Cross-Validation: ", mse, "\n")

# Print the cross-validation scores
print("Cross-Validation Scores:", cv_scores)
print("Mean Squared Error after Cross-Validation:", cv_scores.mean(), "\n")

# Evaluating the model performance
train_accuracy = accuracy_score(spY_train, spY_train_pred)
print("Train Accuracy:", train_accuracy)

test_accuracy = accuracy_score(spY_test, spY_test_pred)
print ("Test Accuracy:", test_accuracy, "\n")

# Displaying classification report
print("Classification Report:\n", classification_report(spY_test, spY_test_pred))

# Visualizing the predictions
plt.figure(figsize=(18, 9))
plt.fill_between(range(len(spY_test)), stock_data["Target"][:len(spY_test)], label='Actual Data')
plt.fill_between(range(len(spY_test)), spY_test, color='red', label='Predicted Data')

# Set the date column to represent actual dates
plt.xticks(range(0, len(spY_test), 100), stock_data['Date'].iloc[:len(spY_test):100], rotation=45)
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close', fontsize=18)
plt.legend()
plt.show()


