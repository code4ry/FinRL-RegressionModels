import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
import matplotlib.pyplot as plt
import random

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

# TimeSeriesSplit with 3 splits
tscv = TimeSeriesSplit(n_splits=3)

# Initialize lists to store results
mse_scores = []
actual_values = []
predicted_values = []
test_dates = []

# Perform Time Series Cross Validation
for train_index, test_index in tscv.split(df_features):
    train_data, test_data = df_features.iloc[train_index], df_features.iloc[test_index]
    
    # Extract features and target variables for training data
    X_train = train_data.drop(['Adj Close', 'Date'], axis=1)
    y_train = train_data['Adj Close']

    # Extract features and target variables for testing data
    X_test = test_data.drop(['Adj Close', 'Date'], axis=1)
    y_test = test_data['Adj Close']
    test_dates.extend(test_data['Date'])  # Capture test dates for plotting
    
    # Generate random alpha values between 0.001 and 0.01
    alpha_values = [round(random.uniform(0.001, 0.01), 3) for _ in range(5)]  # Generate 5 random alpha values
    
    best_alpha = None
    best_mse = float('inf')
    
    # Train Lasso model with different alpha values and select the best one
    for alpha in alpha_values:
        lasso = Lasso(alpha=alpha)
        cv_scores = cross_val_score(lasso, X_train, y_train, cv=3, scoring='neg_mean_squared_error')
        avg_mse = -np.mean(cv_scores)
        
        if avg_mse < best_mse:
            best_mse = avg_mse
            best_alpha = alpha
    
    # Train final Lasso model with the best alpha
    final_lasso = Lasso(alpha=best_alpha)
    final_lasso.fit(X_train, y_train)

    # Predict on testing data
    test_pred = final_lasso.predict(X_test)

    # Calculate MSE on testing data
    test_mse = mean_squared_error(y_test, test_pred)
    mse_scores.append(test_mse)
    actual_values.extend(y_test)
    predicted_values.extend(test_pred)
    
    # Print the final alpha value used for the current split
    print(f"Final Alpha Value for Split: {best_alpha}")

# Print the average testing MSE across splits
print(f"Average Testing MSE: {np.mean(mse_scores)}")

# Plot actual vs predicted for testing data
plt.figure(figsize=(10, 6))
plt.scatter(test_dates, actual_values, color='blue', label='Actual', alpha=0.7)  # Scatter plot for actual values
plt.scatter(test_dates, predicted_values, color='green', label='Predicted', alpha=0.7)  # Scatter plot for predicted values
plt.plot(test_dates, predicted_values, color='green', label='Predicted Line', alpha=0.7)  # Line plot for predicted values
plt.xlabel('Date')
plt.ylabel('Adjusted Closing Score')
plt.title('Actual vs Predicted Adjusted Closing Score (Test Data)')
plt.legend()
plt.show()
