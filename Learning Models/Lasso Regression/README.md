## Organization
The base sub-folder contains three different sub-folders, they are:
   1. alphaTesting
     - This folder contains a file which simulates and tests what the best possible alpha value is for a given dataset
   2. differentIntervals
      - This folder contains several other folders which test the impact of different testing/training intervals
   3. LassoRidge
      - This folder contains the final version of my section of the project. This is what was used to derive the data presented for the RCOS Expo.
        
## Usage
1. Replace the parameter for line 9 with intended dataset for usage.
2. Let's say we want to analyze "dji.csv" located in the "Dataset" sub-folder.
3. Right-click "dji.csv" and select "Copy Relative Path"
4. This pathing should then be pasted into the parameter for line 9
   4a.  This will look like: df = pd.read_csv("Datasets\dji.csv")  

## Lasso-Ridge Regression Model Explanation and Program Analysis
As listed in organization the code for the Lasso-Ridge model is stored in the "LassoRidge" sub-folder.

# Why did I switch from a Lasso Regression model to a Lasso-Ridge Regression Model?
I switched from a Lasso Regression model to a Lasso-Ridge Regression Model because the latter benefits from regularization methods of both Lasso and Ridge, which helps in producing better model performance by effectively managing overfitting. The Lasso-Ridge model combines the strengths of both techniques, allowing for more flexibility in handling multicollinearity and selecting important features while still controlling model complexity. This shift was particularly useful when dealing with stocks as the Lasso alone was too aggressive in feature selection, which is why there was a need for more nuanced regularization

# Explaining sklearn.ElasticNetCV
The sklearn.ElasticNet is a regression model that combines the L1 and L2 penalties of the Lasso and Ridge methods, respectively. It is useful for variable selection and addressing multicollinearity in linear regression.

Key Parameters:
   alpha: Controls the total regularization penalty.
   l1_ratio: The mixing parameter, where l1_ratio = 0 corresponds to Ridge, l1_ratio = 1 corresponds to Lasso, and 0 < l1_ratio < 1 gives a combination of both.
   fit_intercept: Whether to calculate the intercept for the model.
   normalize: If True, the regressors are normalized before fitting.
   
Attributes:
   coef_: Estimated coefficients for the linear regression problem.
   intercept_: Independent term in the linear model.
   Methods:
   fit(X, y): Fit the Elastic Net model.
   predict(X): Predict using the fitted model.
   score(X, y): Returns the coefficient of determination R^2 of the prediction.

# Program Analysis for "lasridge.py"
This Python program demonstrates using an ElasticNet regression model to predict stock closing prices based on historical data. Below is a breakdown of the program's steps and functionalities:

This Python program demonstrates using an ElasticNet regression model to predict stock closing prices based on historical data. Below is a breakdown of the program's steps and functionalities:

1. Import Libraries:
   The program imports necessary libraries including pandas, numpy, matplotlib.pyplot, and modules from sklearn for data manipulation, visualization, and machine learning tasks.

2. Load and Prepare Data:
   - The program loads historical stock data from a CSV file using pd.read_csv().
   - The 'Date' column in the DataFrame is converted to datetime format and set as the index using pd.to_datetime() and set_index() methods.
   - Any rows containing missing values are removed using dropna().

3. Define Features and Target Variable:
   - Features (X) are selected as 'Open', 'High', 'Low', and 'Volume'.
   - Target variable (y) is set as 'Close', representing the closing price of the stock.

4. Split Data into Train and Test Sets:
   - A split date ('2023-01-01') is defined to separate the data into training and test sets based on the index date.
   - Data before the split date is used for training (X_train, y_train), while data on or after the split date is used for testing (X_test, y_test).

5. Feature Scaling:
   - The training and test feature sets (X_train, X_test) are scaled using StandardScaler() to standardize the features by removing the mean and scaling to unit variance.

6. Model Training and Hyperparameter Tuning:
   - An ElasticNetCV model is initialized with 5-fold cross-validation (cv=5) and a specified random state.
   - The model is trained (fit()) on the scaled training data (X_train_scaled, y_train) to automatically determine the best alpha (best_alpha) and l1_ratio (best_l1_ratio) hyperparameters      using cross-validation.

7. Print Best Hyperparameters:
The program prints the best alpha and l1_ratio values selected by the ElasticNetCV model.

8. Initialize and Fit the Best Model:
   - An ElasticNet model is initialized using the best hyperparameters (best_alpha, best_l1_ratio).
   - The best model is trained (fit()) on the entire training set (X_train_scaled, y_train).

9. Model Evaluation:
   - The trained model is used to make predictions (predict()) on the scaled test set (X_test_scaled).
   - Mean Squared Error (MSE) and R-squared (R2) metrics are calculated using mean_squared_error() and r2_score() comparing the predicted values (y_pred_test) with the actual values    
     (y_test).

10. Visualization:
A plot is generated using matplotlib.pyplot to visualize the actual closing prices (y_test) versus the predicted closing prices (y_pred_test) for the test data.


