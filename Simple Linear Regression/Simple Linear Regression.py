# 📌 Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 📥 Load the dataset
df = pd.read_csv("Salary_Data.csv")

# 🎯 Separate features and target variable
X = df.iloc[:, 0].values.reshape(-1, 1)  # Years of Experience
y = df.iloc[:, 1].values.reshape(-1, 1)  # Salary

# ✂️ Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 🧠 Train the Linear Regression model
reg = LinearRegression()
reg.fit(X_train, y_train)

# 📈 Predict on test data
y_pred = reg.predict(X_test)

# 📊 Model coefficients
slope = reg.coef_[0][0]
intercept = reg.intercept_[0]
print(f"Regression Equation: y = {slope:.2f} * X + {intercept:.2f}")

# 📌 Predict salary for 12 years of experience (example)
example_years = 12
predicted_salary = reg.predict([[example_years]])[0][0]
print(f"Predicted salary for {example_years} years of experience: ₹{predicted_salary:.2f}")

# 📏 Error Metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n📊 Model Evaluation Metrics:")
print(f"MAE:  ₹{mae:.2f}")
print(f"MSE:  ₹{mse:.2f}")
print(f"RMSE: ₹{rmse:.2f}")
print(f"R² Score: {r2:.4f}")

# 📉 Visualization
plt.figure(figsize=(8, 5))
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, reg.predict(X), color='red', linewidth=2, label='Best Fit Line')
plt.xlabel('Years of Experience')
plt.ylabel('Salary (₹)')
plt.title('Simple Linear Regression: Experience vs Salary')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
