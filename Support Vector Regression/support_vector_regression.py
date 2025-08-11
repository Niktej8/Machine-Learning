"""
Support Vector Regression (SVR) with Epsilon-Tube Visualization
Dataset: Position_Salaries.csv
Author: [Your Name]
Description:
    - Trains an SVR model with polynomial kernel
    - Scales both features (X) and target (y)
    - Visualizes predictions and the epsilon-tube
"""

# ===============================
# 📦 Import Libraries
# ===============================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ===============================
# 📂 Load Dataset
# ===============================
df = pd.read_csv("Position_Salaries.csv")

# Features (X) and target (y)
X = df[['Level']].values
y = df['Salary'].values  # 1D array

# ===============================
# ✂️ Train-Test Split
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# ===============================
# ⚖️ Feature Scaling
# ===============================
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

# y must be reshaped for scaling, then flattened
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).ravel()

# ===============================
# 🤖 Train SVR Model
# ===============================
regressor = SVR(kernel='poly', epsilon=0.1, degree=3)
regressor.fit(X_train_scaled, y_train_scaled)

# ===============================
# 📈 Predictions
# ===============================
y_pred_scaled = regressor.predict(X_test_scaled)
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

# ===============================
# 📊 Model Evaluation
# ===============================
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n📊 Model Evaluation Metrics:")
print(f"MAE:  ₹{mae:.2f}")
print(f"MSE:  ₹{mse:.2f}")
print(f"RMSE: ₹{rmse:.2f}")
print(f"R² Score: {r2:.4f}")

# ===============================
# 📉 Visualization: Epsilon Tube
# ===============================
# Create smooth X grid
X_grid = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)
X_grid_scaled = scaler_X.transform(X_grid)

# Predictions on grid
y_grid_scaled = regressor.predict(X_grid_scaled)
y_grid = scaler_y.inverse_transform(y_grid_scaled.reshape(-1, 1)).ravel()

# Epsilon boundaries
epsilon = regressor.epsilon
y_upper_scaled = y_grid_scaled + epsilon
y_lower_scaled = y_grid_scaled - epsilon

# Inverse transform boundaries
y_upper = scaler_y.inverse_transform(y_upper_scaled.reshape(-1, 1)).ravel()
y_lower = scaler_y.inverse_transform(y_lower_scaled.reshape(-1, 1)).ravel()

# Plot
plt.figure(figsize=(8,6))
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X_grid, y_grid, color='red', label='SVR Prediction')
plt.plot(X_grid, y_upper, 'g--', label='Epsilon Upper')
plt.plot(X_grid, y_lower, 'g--', label='Epsilon Lower')
plt.fill_between(X_grid.ravel(), y_lower, y_upper, color='green', alpha=0.1)
plt.xlabel('Level')
plt.ylabel('Salary')
plt.title('Support Vector Regression (Epsilon-Tube)')
plt.legend()
plt.show()
