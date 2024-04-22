import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv('Learning Models\Logistic Regression\logisticRegression\TSLA - Full.csv')

# Consider only relevant columns (features and target)
X = df[['Open', 'High', 'Low', 'Adj Close', 'Volume']]

# Target variable: Percentage change greater than 1% as positive (1)
threshold = 0.01
y = np.where(df['Close'].pct_change() > threshold, 1, 0)

# Define time series split
tscv = TimeSeriesSplit(n_splits=5)  # Number of splits for time series cross-validation

# Iterate over each time series split
for train_index, test_index in tscv.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Initialize and train the logistic regression model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Predict using the trained model
    y_pred = model.predict(X_test)

    # Evaluate the model on the test data
    accuracy = model.score(X_test, y_test)
    auc_roc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

    print(f"Model Accuracy: {accuracy}")
    print(f"AUC-ROC: {auc_roc}")

    # Plot predicted probability vs closing price (instead of S-curve)
    plt.figure(figsize=(8, 6))
    predicted_proba = model.predict_proba(X_test)[:, 1]
    plt.scatter(df['Close'].iloc[test_index], predicted_proba, color='blue', label='Predicted Increase Probability')
    plt.xlabel('Closing Value')
    plt.ylabel('Predicted Increase Probability')
    plt.title('Logistic Regression: Predicted Increase Probability (Test Data)')
    plt.legend()
    plt.show()

    # Plot Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()

    # Plot Feature Importance
    feature_importance = pd.Series(model.coef_[0], index=X.columns)
    feature_importance.plot(kind='barh', figsize=(8, 6
