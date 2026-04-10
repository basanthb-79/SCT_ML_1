# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score

# Load training and test datasets
train_df = pd.read_csv("house_prices_train.csv")
test_df = pd.read_csv("house_prices_test.csv")

# Preview the training dataset
print("Training dataset sample:\n", train_df.head())

# Choose relevant features (adjust if dataset has different names)
features = [
    'MSSubClass', 'LotArea', 'OverallQual', 'OverallCond',
    'YearBuilt', 'GrLivArea', 'GarageCars', 'GarageArea',
    'FullBath', 'HalfBath', 'TotRmsAbvGrd', 'YearRemodAdd'
]

# Separate predictors and target variable
X = train_df[features]
y = train_df['SalePrice']

# Handle missing values with mean imputation
imputer = SimpleImputer(strategy="mean")
X_imputed = imputer.fit_transform(X)

# Apply same transformation to test data
X_test = test_df[features]
X_test_imputed = imputer.transform(X_test)

# Initialize and train the regression model
lin_reg = LinearRegression()
lin_reg.fit(X_imputed, y)

# Evaluate model on training set 
train_preds = lin_reg.predict(X_imputed)
print("\nTraining Performance:")
print("MSE:", mean_squared_error(y, train_preds))
print("R²:", r2_score(y, train_preds))

# Predict prices for test dataset
test_preds = lin_reg.predict(X_test_imputed)

# Build submission DataFrame
submission_df = pd.DataFrame({
    "Id": test_df["Id"],
    "SalePrice": test_preds
})

# Save submission file
submission_df.to_csv("submission.csv", index=False)
print("\nSubmission file created: submission.csv")

# visualize training predictions
plt.figure(figsize=(8,6))
plt.scatter(y_train, train_preds, color="lightblue", alpha=0.6)
plt.plot([y_train.min(), y_train.max()],
         [y_train.min(), y_train.max()],
         color="red", linewidth=2,linestyle = "--")  # perfect prediction line
plt.xlabel("Actual SalePrice")
plt.ylabel("Predicted SalePrice")
plt.title("Training Set: Actual vs Predicted Prices")
plt.grid(True)

plt.show()

