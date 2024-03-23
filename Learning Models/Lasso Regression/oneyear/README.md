
## Usage

1. **Dataset**: Replace `'FinRL-RegressionModels\Datasets\sandp500.csv'` with the actual filename of your dataset containing historical stock market data.

2. **Data Preprocessing**:
   - Convert the 'Date' column to datetime format.
   - Filter data for training and testing based on specified date ranges ('2022-01-01' to '2022-12-31' for training, '2023-01-01' to '2023-12-31' for testing).
   - Define time series features using lagged values of the adjusted closing score.

3. **Model Training**:
   - Extract features and target variables from the training and testing datasets.
   - Train the Lasso regression model with specified alpha parameter (adjustable for controlling regularization strength).

4. **Model Evaluation**:
   - Predict adjusted closing scores on both training and testing datasets.
   - Calculate Mean Squared Error (MSE) for training and testing predictions.
   - Plot actual vs. predicted adjusted closing scores for testing data to visually assess model performance.

## Example Output

