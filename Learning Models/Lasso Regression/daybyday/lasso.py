import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error

# Load dataset
df = pd.read_csv('FinRL-RegressionModels\Datasets\sandp500.csv')  # Replace 'your_dataset.csv' with the actual filename

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

# Initialize lists to store results
train_mses = []
test_mses = []

# Iterate over each day of 2022 for training
for day in pd.date_range(start='2022-01-01', end='2022-12-31'):
    # Filter data for the current day
    train_day_data = train_data[train_data['Date'] <= day]
    
    # Create features for training data
    train_day_data = create_features(train_day_data)
    
    # Extract features and target variables
    X_train = train_day_data.drop(['Adj Close', 'Date'], axis=1)
    y_train = train_day_data['Adj Close']
    
    # Train Lasso model
    lasso = Lasso(alpha=0.001)  # Adjust alpha parameter to control regularization strength
    lasso.fit(X_train, y_train)
    
    # Filter test data for the next day
    next_day = day + pd.Timedelta(days=1)
    test_day_data = test_data[test_data['Date'] == next_day]
    
    if not test_day_data.empty:
        # Create features for test data
        test_day_data = create_features(test_day_data)
    
        # Extract features and target variables for test data
        X_test = test_day_data.drop(['Adj Close', 'Date'], axis=1)
        y_test = test_day_data['Adj Close']
    
        # Predict on test data
        test_pred = lasso.predict(X_test)

        # Calculate MSE on testing data
        test_mse = mean_squared_error(y_test, test_pred)
        
        # Append MSE to the list
        test_mses.append(test_mse)
    
    # Print the result for the current day
    print(f"Trained until: {day.strftime('%Y-%m-%d')} - Testing MSE: {test_mse}")

# Calculate average MSE over all test days
avg_test_mse = np.mean(test_mses)
print(f"Average Testing MSE: {avg_test_mse}")
