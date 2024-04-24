import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('Datasets/sandp500.csv')  

# Convert 'Date' column to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Filter data for training and testing
train_data = df[(df['Date'] >= '2022-01-01') & (df['Date'] <= '2022-12-31')]
test_data = df[(df['Date'] >= '2023-01-01') & (df['Date'] <= '2023-12-31')]

# Slice is a "refrence", make copy instead
def create_features(data):
    data_copy = data.copy()
    data_copy.loc[:, 'lag_1'] = data_copy['Adj Close'].shift(1)
    data_copy.loc[:, 'lag_2'] = data_copy['Adj Close'].shift(2)
    data_copy.loc[:, 'lag_3'] = data_copy['Adj Close'].shift(3)
    return data_copy.dropna()


# Create features for training and testing data
train_data = create_features(train_data)
test_data = create_features(test_data)

# Extract features and target variables
X_train = train_data.drop(['Adj Close', 'Date'], axis=1)
y_train = train_data['Adj Close']

X_test = test_data.drop(['Adj Close', 'Date'], axis=1)
y_test = test_data['Adj Close']

# Train Lasso model
lasso = Lasso(alpha=0.001)  # Adjust alpha parameter to control regularization strength
lasso.fit(X_train, y_train)

# Predict on training data
train_pred = lasso.predict(X_train)

# Predict on testing data
test_pred = lasso.predict(X_test)

# Calculate MSE on training and testing data
train_mse = mean_squared_error(y_train, train_pred)
test_mse = mean_squared_error(y_test, test_pred)

# Plot actual vs predicted for testing data
plt.figure(figsize=(10, 6))
plt.scatter(y_test, test_pred, color='blue', label='Actual vs Predicted')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, color='red', label='Ideal')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted Adjusted Closing Score (Testing Data)')
plt.legend()
plt.show()

print(f"Training MSE: {train_mse}")
print(f"Testing MSE: {test_mse}")