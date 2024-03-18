import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Read the dataset from CSV file
df = pd.read_csv("Datasets\sandp500.csv")

# Convert the 'Date' column to datetime type
df['Date'] = pd.to_datetime(df['Date'])

# Get unique years from the dataset
unique_years = df['Date'].dt.year.unique()
# Print the available years to the user
print(f"Available Years: \n{unique_years}")
# Ask the user to input the year
selected_year = int(input("Enter the year you want to analyze: "))
# Filter the dataset for the selected year
df_selected_year = df[df['Date'].dt.year == selected_year]

# Get unique months from the selected year in dataset
unique_months = df_selected_year['Date'].dt.month.unique()
print(f"Avaliabe Months: \n{unique_months}")
#Ask the user to input month they want to analyze from selected year
selected_month = int(input("Enter the month you want to analyze (1-12): "))

# Filter the dataset for the selected year and month
df_selected = df[(df['Date'].dt.year == selected_year) & (df['Date'].dt.month == selected_month)]

# Extract day of the week (1 for Monday, 2 for Tuesday, ..., 7 for Sunday)
df_selected['Day_of_Week'] = df_selected['Date'].dt.dayofweek + 1

# Iterate over each day of the week
for day in range(1, 8):  # Days of the week range from 1 (Monday) to 7 (Sunday)
    # Filter the dataset for the current day
    df_day = df_selected[df_selected['Day_of_Week'] == day]

    # Split the dataset into features (X) and target variable (y)
    X_day = df_day[['Day_of_Week']]
    y_day = df_day['Adj Close']

    # Determine the number of samples in the dataset
    num_samples = len(df_selected_year) + 1

    # Adjust the number of folds for cross-validation if necessary
    n_splits = min(5, num_samples)  # Set the number of folds to a maximum of 5 or the number of samples, whichever is smaller

    # Initialize TimeSeriesSplit with the adjusted number of splits
    tscv = TimeSeriesSplit(n_splits=n_splits)

    # Initialize lists to store evaluation metrics for each split
    mse_per_split = []
    rsquared_per_split = []

    # Iterate over splits
    for train_index, test_index in tscv.split(X_day):
        X_train, X_test = X_day.iloc[train_index], X_day.iloc[test_index]
        y_train, y_test = y_day.iloc[train_index], y_day.iloc[test_index]

        # Train a Lasso regression model
        lasso = Lasso(alpha=0.1)
        lasso.fit(X_train, y_train)

        # Make predictions
        y_pred = lasso.predict(X_test)

        # Calculate evaluation metrics
        mse = mean_squared_error(y_test, y_pred)
        mse_per_split.append(mse)

        rsquared = r2_score(y_test, y_pred)
        rsquared_per_split.append(rsquared)

        # Plot actual vs. predicted values for the current split
        plt.figure(figsize=(6, 4))
        plt.scatter(X_test, y_test, label='Actual Adj Close Price', color='blue', alpha=0.5)
        plt.scatter(X_test, y_pred, label='Predicted Adj Close Price', color='red', alpha=0.5)
        plt.xlabel('Day of the Week')
        plt.ylabel('Adjusted Close Price')
        plt.title(f'Actual vs. Predicted Adj Close Price for Day {day}, Split {len(mse_per_split)}')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
