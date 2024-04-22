import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load dataset (change file path as needed)
df = pd.read_csv('Learning Models\Lasso Regression\sandp500.csv')

# Convert 'Date' column to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Define features (X) and target variable (y)
X = df[['Open', 'High', 'Low', 'Volume']]
y = df['Close']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Initialize ElasticNetCV with alpha range and L1 ratio
alphas = np.linspace(0.1, 10, 100)  # Range of alpha values to search over
model = ElasticNetCV(alphas=alphas, l1_ratio=0.5, cv=5)

# Fit the model
model.fit(X_train, y_train)

# Get the best alpha value
best_alpha = model.alpha_

# Make predictions on the test set using the best model
y_pred = model.predict(X_test)

# Calculate Mean Squared Error (MSE) and R-squared (R2) on the test set
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Display model evaluation metrics
print(f'Best Alpha: {best_alpha}')
print(f'Mean Squared Error (MSE) on test set: {mse}')
print(f'R-squared (R2) on test set: {r2}')

# Visualize actual vs. predicted closing prices for test data
plt.figure(figsize=(10, 6))
plt.plot(df['Date'].iloc[-len(y_test):], y_test, label='Actual Close', marker='o')
plt.plot(df['Date'].iloc[-len(y_test):], y_pred, label='Predicted Close', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.title('Actual vs. Predicted Closing Prices (Test Data)')
plt.legend()
plt.grid(True)
plt.show()
