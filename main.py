
# importing required libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# loading the dataset
df = pd.read_csv("LifeExpectancyData.csv")


df.columns = df.columns.str.strip()

# drop colums that don't contribute to prediction
columns_to_drop = ['Country', 'Year', 'Status']
df = df.drop(columns=[col for col in columns_to_drop if col in df.columns], errors='ignore')

# checking if there's any missing values
print(df.isnull().sum())

# droping rows where life expectancy is empty as cannot train with these
df = df[df['Life expectancy'].notnull()]

# fill other missing values with column means
df = df.fillna(df.mean(numeric_only=True))

# checking if that worked
print(df.isnull().sum())

# separating featues and target
X = df.drop('Life expectancy', axis = 1) # inputs
y = df['Life expectancy']                # target

# splitting data into training and temp (validation + test) sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=90) # 0.8 for training, 0.2 for temp (80%, 20%)

X_vali, X_test, y_vali, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=90) # 0.5 of 0.2 for each validating and testing (10%)


# checking if df contains non-numeric fields and encoding if it does
print(df.dtypes)

# initializing the model
model = RandomForestRegressor(
    n_estimators=200,    # no.of trees (default 100)
    max_depth=10,        # max depth of each tree (default is None i.e. grow until pure)
    min_samples_split=5, # minimum samples needed to split a node (default 2)
    random_state=42      # for reproducibility
)

# training the model on the training set
model.fit(X_train, y_train)

# predicting and evaluating
y_pred = model.predict(X_vali)

# evaluation of the model using MAE, MSE, R squared
mae = mean_absolute_error(y_vali, y_pred)
mse = mean_squared_error(y_vali, y_pred)
r2 = r2_score(y_vali, y_pred)

print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("R² Score:", r2)

# predicting on the test set
y_test_pred = model.predict(X_test)

# final performance
mae_test = mean_absolute_error(y_test, y_test_pred)
mse_test = mean_squared_error(y_test, y_test_pred)
r2_test = r2_score(y_test, y_test_pred)
print("MODEL PERFORMANCE:")
print("Test MAE:", mae_test)
print("Test MSE:", mse_test)
print("Test R² Score:", r2_test)


# visualization
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_test_pred, alpha=0.6, color='teal')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # Perfect line
plt.xlabel("Actual Life Expectancy")
plt.ylabel("Predicted Life Expectancy")
plt.title("Actual vs Predicted Life Expectancy")
plt.grid(True)
plt.tight_layout()
plt.show()

# residuals
residuals = y_test - y_test_pred

plt.figure(figsize=(8, 5))
plt.scatter(y_test_pred, residuals, color='purple', alpha=0.5)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Predicted Life Expectancy")
plt.ylabel("Residuals (Actual - Predicted)")
plt.title("Residuals vs Predicted Values")
plt.grid(True)
plt.tight_layout()
plt.show()


# feature importance
importances = model.feature_importances_
feature_names = X.columns
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.bar(range(len(importances)), importances[indices], align='center', color='royalblue')
plt.xticks(range(len(importances)), feature_names[indices], rotation=90)
plt.title("Feature Importances")
plt.tight_layout()
plt.show()


