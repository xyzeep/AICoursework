
# importing required libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


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
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42) # 0.8 for training, 0.2 for temp (80%, 20%)

X_vali, X_test, y_vali, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42) # 0.5 of 0.2 for each validating and testing (10%)


# checking if df contains non-numeric fields and encoding if it does
print(df.dtypes)

# initializing the model
model = RandomForestRegressor(random_state=42)

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


