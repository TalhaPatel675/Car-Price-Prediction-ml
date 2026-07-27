import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ---------------------------------
# Create folders
# ---------------------------------
os.makedirs("models", exist_ok=True)
os.makedirs("images", exist_ok=True)

# ---------------------------------
# Load dataset
# ---------------------------------
df = pd.read_csv("data/CarPrice_Assignment.csv")

# Remove unnecessary columns
df.drop(columns=["car_ID", "CarName"], errors="ignore", inplace=True)

# ---------------------------------
# Features and Target
# ---------------------------------
X = df.drop("price", axis=1)
y = df["price"]

# ---------------------------------
# Categorical and Numerical Columns
# ---------------------------------
categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
numerical_features = X.select_dtypes(exclude=["object"]).columns.tolist()

# ---------------------------------
# Preprocessing
# ---------------------------------
preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ],
    remainder="passthrough"
)

# ---------------------------------
# Create Pipeline
# ---------------------------------
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ]
)

# ---------------------------------
# Train-Test Split
# ---------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# ---------------------------------
# Train Model
# ---------------------------------
model.fit(X_train, y_train)

# ---------------------------------
# Predictions
# ---------------------------------
predictions = model.predict(X_test)

# ---------------------------------
# Evaluation
# ---------------------------------
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

# ---------------------------------
# Results
# ---------------------------------
print("\n" + "=" * 60)
print("🚗 CAR PRICE PREDICTION MODEL")
print("=" * 60)

print(f"Dataset Size : {len(df)}")
print("Model Used   : Linear Regression")
print(f"MAE          : {mae:.2f}")
print(f"RMSE         : {rmse:.2f}")
print(f"R² Score     : {r2:.4f}")

# ---------------------------------
# Save Model
# ---------------------------------
joblib.dump(model, "models/car_price_pipeline.pkl")

print("\n✅ Model saved successfully!")

# ---------------------------------
# Actual vs Predicted Plot
# ---------------------------------
plt.figure(figsize=(8,6))
plt.scatter(y_test, predictions)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Prices")
plt.grid(True)

plt.savefig("images/actual_vs_predicted.png")
plt.close()

# ---------------------------------
# Residual Plot
# ---------------------------------
residuals = y_test - predictions

plt.figure(figsize=(8,6))
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