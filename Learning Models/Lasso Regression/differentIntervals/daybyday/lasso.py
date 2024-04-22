import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('Datasets\sandp500.csv')  # Replace 'your_dataset.csv' with the actual filename

# Convert 'Date' column to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Create features with lagged values
def create_features(data, lag_features=3):
    data_copy = data.copy()
    for i in range(1, lag_features + 1):
        data_copy[f'lag_{i}'] = data_copy['Adj Close'].shift(i)
    return data_copy.dropna()

# Create lagged features for the entire dataset
df_features = create_features(df)

# TimeSeriesSplit with 2 splits
tscv = TimeSeriesSplit(n_splits=2)

# Initialize lists to store results
mse_scores = []
actual_values = []
predicted_values = []

# Perform Time Series Cross Validation
for train_index, test_index in tscv.split(df_features):
    train_data, test_data = df_features.iloc[train_index], df_features.iloc[test_index]
    
    # Extract features and target variables for training data
    X_train = train_data.drop(['Adj Close', 'Date'], axis=1)
    y_train = train_data['Adj Close']

    # Extract features and target variables for testing data
    X_test = test_data.drop(['Adj Close', 'Date'], axis=1)
    y_test = test_data['Adj Close']
    
    # Train Lasso model
    lasso = Lasso(alpha=0.001)
    lasso.fit(X_train, y_train)

    # Predict on testing data
    test_pred = lasso.predict(X_test)

    # Calculate MSE on testing data
    test_mse = mean_squared_error(y_test, test_pred)
    mse_scores.append(test_mse)
    actual_values.extend(y_test)
    predicted_values.extend(test_pred)

# Print the average testing MSE
print(f"Average Testing MSE: {np.mean(mse_scores)}")

# Plot actual vs predicted for testing data
plt.figure(figsize=(10, 6))
plt.plot(df_features['Date'], df_features['Adj Close'], color='blue', label='Actual', alpha=0.7)
plt.plot(df_features['Date'].iloc[-len(predicted_values):], predicted_values, color='green', label='Predicted', alpha=0.7)
plt.xlabel('Date')
plt.ylabel('Adjusted Closing Score')
plt.title('Actual vs Predicted Adjusted Closing Score')
plt.legend()
plt.show()
