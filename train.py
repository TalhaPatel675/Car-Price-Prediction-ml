import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

# ---------------------------------
# Create required folders
# ---------------------------------
os.makedirs("models", exist_ok=True)
os.makedirs("images", exist_ok=True)

# ---------------------------------
# Load Dataset
# ---------------------------------
df = pd.read_csv("data/CarPrice_Assignment.csv")

# ---------------------------------
# Data Preprocessing
# ---------------------------------

# Remove unnecessary columns
df.drop(columns=["car_ID", "CarName"], errors="ignore", inplace=True)

# Convert categorical columns into numerical values
df = pd.get_dummies(df, drop_first=True)

# ---------------------------------
# Features and Target
# ---------------------------------
X = df.drop("price", axis=1)
y = df["price"]

# ---------------------------------
# Train-Test Split
# ---------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
)

# ---------------------------------
# Train Model
# ---------------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# ---------------------------------
# Make Predictions
# ---------------------------------
predictions = model.predict(X_test)

# ---------------------------------
# Evaluate Model
# ---------------------------------
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, predictions)

# ---------------------------------
# Display Results
# ---------------------------------
print("\n" + "=" * 60)
print("🚗 CAR PRICE PREDICTION MODEL")
print("=" * 60)

print(f"Dataset Size : {len(df)}")
print(f"Model Used   : Linear Regression")
print(f"MAE          : {mae:.2f}")
print(f"RMSE         : {rmse:.2f}")
print(f"R² Score     : {r2:.4f}")

# ---------------------------------
# Save Model
# ---------------------------------
joblib.dump(model, "models/linear_regression.pkl")

print("\n✅ Model saved successfully!")

# ---------------------------------
# Actual vs Predicted Plot
# ---------------------------------
plt.figure(figsize=(8, 6))
plt.scatter(y_test, predictions)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Car Prices")
plt.grid(True)

plt.savefig("images/actual_vs_predicted.png")
plt.close()

# ---------------------------------
# Residual Plot
# ---------------------------------
residuals = y_test - predictions

plt.figure(figsize=(8, 6))
plt.scatter(predictions, residuals)
plt.axhline(y=0, color="red", linestyle="--")
plt.xlabel("Predicted Price")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.grid(True)

plt.savefig("images/residual_plot.png")
plt.close()

print("✅ Graphs saved successfully!")

print("\nProject completed successfully!")
