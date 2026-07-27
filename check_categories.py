import pandas as pd

df = pd.read_csv("data/CarPrice_Assignment.csv")

categorical_columns = [
    "fueltype",
    "aspiration",
    "doornumber",
    "carbody",
    "drivewheel",
    "enginelocation",
    "enginetype",
    "cylindernumber",
    "fuelsystem"
]

for col in categorical_columns:
    print(f"\n{col}:")
    print(sorted(df[col].unique()))
