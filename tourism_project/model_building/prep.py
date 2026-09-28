
"""Load the dataset, clean it, and save train/test splits to the repo root."""    main()

import pandas as pdif __name__ == "__main__":

from sklearn.model_selection import train_test_split



DATA_PATH = "tourism_project/data/tourism.csv"    print(f"Train shape: {Xtrain.shape}, Test shape: {Xtest.shape}")



    ytest.to_csv("ytest.csv", index=False)

def main():    ytrain.to_csv("ytrain.csv", index=False)

    df = pd.read_csv(DATA_PATH)    Xtest.to_csv("Xtest.csv", index=False)

    Xtrain.to_csv("Xtrain.csv", index=False)

    # Drop identifier / index columns that carry no predictive signal

    drop_cols = [c for c in ["Unnamed: 0", "CustomerID"] if c in df.columns]    )

    df = df.drop(columns=drop_cols)        X, y, test_size=0.2, random_state=42, stratify=y

    Xtrain, Xtest, ytrain, ytest = train_test_split(

    df = df.dropna(subset=["ProdTaken"])

    y = df["ProdTaken"]
    X = df.drop(columns=["ProdTaken"])
