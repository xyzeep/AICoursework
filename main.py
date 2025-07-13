# importing required libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib

# loading the dataset
df = pd.read_csv("LifeExpectancyData.csv")

df.columns = df.columns.str.strip()

# drop colums that don't contribute to prediction
columns_to_drop = ["Country", "Year", "Status"]
df = df.drop(
    columns=[col for col in columns_to_drop if col in df.columns], errors="ignore"
)

# checking if there's any missing values
print(df.isnull().sum())

# droping rows where life expectancy is empty as cannot train with these
df = df[df["Life expectancy"].notnull()]

# fill other missing values with column means
df = df.fillna(df.mean(numeric_only=True))

# checking if that worked
print(df.isnull().sum())

# saving the cleaned dataset as I might need it
df.to_csv("cleaned_life_expectancy_data.csv", index=False)

# separating featues and target
X = df.drop("Life expectancy", axis=1)  # inputs
y = df["Life expectancy"]  # target

# splitting data into training and temp (validation + test) sets
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.2, random_state=90
)  # 0.8 for training, 0.2 for temp (80%, 20%)

X_vali, X_test, y_vali, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=90
)  # 0.5 of 0.2 for each validating and testing (10%)


# checking if df contains non-numeric fields and encoding if it does
print(df.dtypes)

# initializing the model
model = RandomForestRegressor(
    n_estimators=200,  # no.of trees (default 100)
    max_depth=10,  # max depth of each tree (default is None i.e. grow until pure)
    min_samples_split=5,  # minimum samples needed to split a node (default 2)
    random_state=90,  # for reproducibility
)

# training the model on the training set
model.fit(X_train, y_train)
# saving the model
joblib.dump(model, "rf_model.pkl")  # save the model

# saving test data to CSV for visualization file
X_test.to_csv("X_test.csv", index=False)
y_test.to_csv("y_test.csv", index=False)

# evaluating
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

# saving predictions for visualization or later use
pd.DataFrame(y_test_pred, columns=["Predicted"]).to_csv("y_pred_test.csv", index=False)

# final performance
mae_test = mean_absolute_error(y_test, y_test_pred)
mse_test = mean_squared_error(y_test, y_test_pred)
r2_test = r2_score(y_test, y_test_pred)
print("MODEL PERFORMANCE:")
print("Test MAE:", mae_test)
print("Test MSE:", mse_test)
print("Test R² Score:", r2_test)
