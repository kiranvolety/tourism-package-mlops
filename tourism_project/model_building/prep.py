"""Load the dataset, clean it, and save train/test splits to the repo root."""
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = "tourism_project/data/tourism.csv"


def main():
    df = pd.read_csv(DATA_PATH)

    # Drop identifier / index columns that carry no predictive signal
    drop_cols = [c for c in ["Unnamed: 0", "CustomerID"] if c in df.columns]
    df = df.drop(columns=drop_cols)

    df = df.dropna(subset=["ProdTaken"])

    X = df.drop(columns=["ProdTaken"])
    y = df["ProdTaken"]

    Xtrain, Xtest, ytrain, ytest = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    Xtrain.to_csv("Xtrain.csv", index=False)
    Xtest.to_csv("Xtest.csv", index=False)
    ytrain.to_csv("ytrain.csv", index=False)
    ytest.to_csv("ytest.csv", index=False)

    print(f"Train shape: {Xtrain.shape}, Test shape: {Xtest.shape}")


if __name__ == "__main__":
    main()
