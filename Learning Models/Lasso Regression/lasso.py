import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Read the dataset from CSV file
# MAKE SURE TO COPY RELATIVE PATH OF FILE BEFORE TESTING EVERY SINGLE TIME
df = pd.read_csv("Datasets\sandp500.csv")

# Convert the 'Date' column to datetime type
df['Date'] = pd.to_datetime(df['Date'])

# Get unique years from the dataset
unique_years = df['Date'].dt.year.unique()
print("Unique Years in the Dataset:")
print(unique_years)

# Ask the user to input the year they want to analyze
selected_year = int(input("Enter the year you want to analyze: "))

# Filter the dataset for the selected year
df_selected_year = df[df['Date'].dt.year == selected_year]

# Get unique months for the selected year
unique_months = df_selected_year['Date'].dt.month.unique()
print(f"Months available in {selected_year}:")
print(unique_months)

# Ask the user to input the month from which year they want to analyze

selected_month = int(input("Enter the month you want to analyze (1-12): "))

# Filter the dataset for the selected month
df_selected_month = df[df['Date'].dt.month == selected_month]

# Extract day of the week (1 for Monday, 2 for Tuesday, ..., 7 for Sunday)
df_selected_month['Day_of_Week'] = df_selected_month['Date'].dt.dayofweek + 1

# Split the dataset into features (X) and target variable (y)
X = df_selected_month[['Day_of_Week']]  # Use only the day of the week as features
y = df_selected_month['Adj Close']  # Predict the 'Adj Close' price

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Lasso regression model
lasso = Lasso(alpha=0.1)  # Adjust the alpha parameter as needed
lasso.fit(X_train, y_train)

# Make predictions
y_pred = lasso.predict(X_test)

# Compare predictions with actual values
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Plot actual vs. predicted values
plt.figure(figsize=(10, 6))
plt.scatter(X_test['Day_of_Week'], y_test.values, label='Actual Adj Close Price', color='blue', alpha=0.5)
plt.scatter(X_test['Day_of_Week'], y_pred, label='Predicted Adj Close Price', color='red', alpha=0.5)
plt.xlabel('Day of the Week (1: Monday, 2: Tuesday, ..., 7: Sunday)')
plt.ylabel('Adjusted Close Price')
plt.title(f'Actual vs. Predicted Adjusted Close Price for Month {selected_month}')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
