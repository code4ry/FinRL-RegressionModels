import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.linear_model import ElasticNetCV, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('Learning Models/Lasso Regression/sandp500.csv')

# Convert 'Date' column to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Ensure no missing values in the dataset
df.dropna(inplace=True)

# Define features (X) and target variable (y)
X = df[['Open', 'High', 'Low', 'Volume']]
y = df['Close']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train the ElasticNetCV model
elastic_net_cv = ElasticNetCV(cv=5, random_state=1)
elastic_net_cv.fit(X_train_scaled, y_train)

# Get the best alpha and l1_ratio from the trained ElasticNetCV model
best_alpha = elastic_net_cv.alpha_
best_l1_ratio = elastic_net_cv.l1_ratio_

# Print the best alpha and l1_ratio found
print(f'Best alpha: {best_alpha}')
print(f'Best l1_ratio: {best_l1_ratio}')

# Initialize the best ElasticNet model with the selected hyperparameters
best_model = ElasticNet(alpha=best_alpha, l1_ratio=best_l1_ratio)

# Perform cross-validated predictions
y_pred_cv = cross_val_predict(best_model, X_train_scaled, y_train, cv=5)

# Calculate Mean Squared Error (MSE) and R-squared (R2) on the train set predictions
mse_cv = mean_squared_error(y_train, y_pred_cv)
r2_cv = r2_score(y_train, y_pred_cv)
print(f'Cross-validated Mean Squared Error (MSE) on train set: {mse_cv}')
print(f'Cross-validated R-squared (R2) on train set: {r2_cv}')

# Fit the best model on the entire training set
best_model.fit(X_train_scaled, y_train)

# Make predictions on the test set using the best model
y_pred_test = best_model.predict(X_test_scaled)

# Calculate Mean Squared Error (MSE) and R-squared (R2) on the test set
mse_test = mean_squared_error(y_test, y_pred_test)
r2_test = r2_score(y_test, y_pred_test)
print(f'Mean Squared Error (MSE) on test set: {mse_test}')
print(f'R-squared (R2) on test set: {r2_test}')

# Visualize actual vs. predicted closing prices for test data
plt.figure(figsize=(10, 6))
plt.plot(df['Date'].iloc[-len(y_test):], y_test, label='Actual Close', marker='o')
plt.plot(df['Date'].iloc[-len(y_test):], y_pred_test, label='Predicted Close', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.title('Actual vs. Predicted Closing Prices (Test Data)')
plt.legend()
plt.grid(True)
plt.show()
