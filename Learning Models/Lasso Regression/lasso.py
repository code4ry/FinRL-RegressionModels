import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Read the dataset from CSV file
df = pd.read_csv("Datasets\sandp500.csv")

# Convert the 'Date' column to datetime type
df['Date'] = pd.to_datetime(df['Date'])

# Get unique months from the dataset
unique_months = df['Date'].dt.month.unique()
print("Unique Months in the Dataset:")
print(unique_months)

# Ask the user to input the month they want to analyze
selected_month = int(input("Enter the month you want to analyze (1-12): "))

# Filter the dataset for the selected month
df_selected_month = df[df['Date'].dt.month == selected_month]

# Split the dataset into features (X) and target variable (y)
X = df_selected_month.drop(columns=['Date', 'Close'])  # Use all columns except 'Date' and 'Close' as features
y = df_selected_month['Close']  # Predict the 'Close' price

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a Lasso regression model
lasso = Lasso(alpha=0.1)  # You can adjust the alpha parameter as needed
lasso.fit(X_train_scaled, y_train)

# Make predictions
y_pred = lasso.predict(X_test_scaled)

# Compare predictions with actual values
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Plot actual vs. predicted values
plt.figure(figsize=(10, 6))
plt.plot(y_test.index, y_test.values, label='Actual Close Price', color='blue')
plt.plot(y_test.index, y_pred, label='Predicted Close Price', color='red')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.title(f'Actual vs. Predicted Close Price for Month {selected_month}')
plt.legend()
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()
