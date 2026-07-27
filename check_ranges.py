import pandas as pd

df = pd.read_csv("data/CarPrice_Assignment.csv")

# Remove columns we don't use
df = df.drop(columns=["car_ID", "CarName", "price"])

numeric_columns = df.select_dtypes(exclude=["object"]).columns

for col in numeric_columns:
    print(f"{col}: {df[col].min()} -> {df[col].max()}")