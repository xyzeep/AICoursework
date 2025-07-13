import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np

# load model and test data
model = joblib.load("rf_model.pkl")
X_test = pd.read_csv("X_test.csv")
y_test = pd.read_csv("y_test.csv")

# predict
y_pred = model.predict(X_test)

# --- Visualization ---

# actual vs predicted scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.6, color='teal', edgecolors='k')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)  # perfect prediction line
plt.xlabel("Actual Life Expectancy")
plt.ylabel("Predicted Life Expectancy")
plt.title("Actual vs Predicted Life Expectancy")
plt.grid(True)
plt.tight_layout()
plt.show()

