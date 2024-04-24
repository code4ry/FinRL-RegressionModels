import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('Datasets\sandp500.csv')  

# Convert 'Date' column to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Define a function to create lagged features
def create_features(data):
    data_copy = data.copy()
    data_copy['lag_1'] = data_copy['Adj Close'].shift(1)
    data_copy['lag_2'] = data_copy['Adj Close'].shift(2)
    data_copy['lag_3'] = data_copy['Adj Close'].shift(3)
    return data_copy.dropna()

# Initialize lists to store results
train_mses = []
test_mses = []

# Iterate over each month of 2022
for month in range(1, 13):
    # Filter data for the current month
    train_month_data = df[(df['Date'] >= f'2022-{month:02d}-01') & (df['Date'] <= f'2022-{month:02d}-31')]
    test_month_data = df[(df['Date'] >= f'2023-{month:02d}-01') & (df['Date'] <= f'2023-{month:02d}-31')]
    
    # Create features for training and testing data
    train_month_data = create_features(train_month_data)
    test_month_data = create_features(test_month_data)
    
    # Extract features and target variables
    X_train = train_month_data.drop(['Adj Close', 'Date'], axis=1)
    y_train = train_month_data['Adj Close']

    X_test = test_month_data.drop(['Adj Close', 'Date'], axis=1)
    y_test = test_month_data['Adj Close']
    
    # Train Lasso model
    lasso = Lasso(alpha=0.01)  # Adjust alpha parameter to control regularization strength
    lasso.fit(X_train, y_train)

    # Predict on training data
    train_pred = lasso.predict(X_train)

    # Predict on testing data
    test_pred = lasso.predict(X_test)

    # Calculate MSE on training and testing data
    train_mse = mean_squared_error(y_train, train_pred)
    test_mse = mean_squared_error(y_test, test_pred)
    
    # Append MSEs to the lists
    train_mses.append(train_mse)
    test_mses.append(test_mse)
    
    # Print the results for the current month
    print(f"Month {month}:")
    print(f"Training MSE: {train_mse}")
    print(f"Testing MSE: {test_mse}")
    print("--------------------")

# Plot MSEs over each month
plt.plot(range(1, 13), train_mses, label='Training MSE')
plt.plot(range(1, 13), test_mses, label='Testing MSE')
plt.xlabel('Month')
plt.ylabel('MSE')
plt.title('MSE for Each Month in 2022')
plt.legend()
plt.xticks(range(1, 13))
plt.show()
