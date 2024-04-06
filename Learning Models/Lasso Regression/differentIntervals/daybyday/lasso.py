import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('FinRL-RegressionModels\Datasets\sandp500.csv')  # Replace 'your_dataset.csv' with the actual filename

# Convert 'Date' column to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Slice is a "refrence", make copy instead
def create_features(data):
    data_copy = data.copy()
    data_copy.loc[:, 'lag_1'] = data_copy['Adj Close'].shift(1)
    data_copy.loc[:, 'lag_2'] = data_copy['Adj Close'].shift(2)
    data_copy.loc[:, 'lag_3'] = data_copy['Adj Close'].shift(3)
    return data_copy.dropna()

# Create features for the entire dataset
df_features = create_features(df)

# Split data into training and testing sets
train_data, test_data = train_test_split(df_features, test_size=0.4, shuffle=False)

# Extract features and target variables for training data
X_train = train_data.drop(['Adj Close', 'Date'], axis=1)
y_train = train_data['Adj Close']

# Extract features and target variables for testing data
X_test = test_data.drop(['Adj Close', 'Date'], axis=1)
y_test = test_data['Adj Close']

# Train Lasso model
lasso = Lasso(alpha=0.001)  # Adjust alpha parameter to control regularization strength
lasso.fit(X_train, y_train)

# Predict on testing data
test_pred = lasso.predict(X_test)

# Calculate MSE on testing data
test_mse = mean_squared_error(y_test, test_pred)

# Print the testing MSE
print(f"Testing MSE: {test_mse}")
# Plot actual vs predicted for testing data
plt.figure(figsize=(10, 6))
plt.scatter(y_test, test_pred, color='blue', label='Actual vs Predicted')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, color='red', label='Ideal')
plt.xlabel('Actual Adjusted Closing Score')
plt.ylabel('Predicted Adjusted Closing Score')
plt.title('Actual vs Predicted Adjusted Closing Score (Testing Data)')
plt.legend()
plt.show()