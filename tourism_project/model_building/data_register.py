"""Validate the tourism dataset and print a short summary."""
import os
import sys

import pandas as pd

DATA_PATH = "tourism_project/data/tourism.csv"

EXPECTED_COLUMNS = [
    "CustomerID", "ProdTaken", "Age", "TypeofContact", "CityTier",
    "DurationOfPitch", "Occupation", "Gender", "NumberOfPersonVisiting",
    "NumberOfFollowups", "ProductPitched", "PreferredPropertyStar",
    "MaritalStatus", "NumberOfTrips", "Passport", "PitchSatisfactionScore",
    "OwnCar", "NumberOfChildrenVisiting", "Designation", "MonthlyIncome",
]


def main():
    if not os.path.exists(DATA_PATH):
        print(f"Dataset not found at {DATA_PATH}")
        sys.exit(1)

    df = pd.read_csv(DATA_PATH)
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        print(f"Missing expected columns: {missing}")
        sys.exit(1)

    print("Dataset registered successfully.")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Target distribution:\n{df['ProdTaken'].value_counts()}")
    print(f"Missing values per column:\n{df.isnull().sum()}")


if __name__ == "__main__":
    main()
