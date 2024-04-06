import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error

# Load dataset
df = pd.read_csv('FinRL-RegressionModels\Datasets\sandp500.csv')  # Replace 'your_dataset.csv' with the actual filename

# Convert 'Date' column to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Define time series features
def create_features(data):
    data['lag_1'] = data['Adj Close'].shift(1)
    data['lag_2'] = data['Adj Close'].shift(2)
    data['lag_3'] = data['Adj Close'].shift(3)
    return data.dropna()

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

# Define a list of alpha values to test
alpha_values = np.arange(0.001, 0.1, 0.001)

# Initialize a dictionary to store results
alpha_results = {}

# Iterate over each alpha value
for alpha in alpha_values:
    # Train Lasso model with current alpha value
    lasso = Lasso(alpha=alpha)
    lasso.fit(X_train, y_train)
    
    # Predict on testing data
    test_pred = lasso.predict(X_test)
    
    # Calculate MSE on testing data
    test_mse = mean_squared_error(y_test, test_pred)
    
    # Store the testing MSE in the results dictionary
    alpha_results[alpha] = test_mse

# Display results in a concise table
print("Alpha\t\tTesting MSE")
print("-----------------------------------")
for alpha, mse in alpha_results.items():
    print(f"{alpha}\t\t{mse}")
