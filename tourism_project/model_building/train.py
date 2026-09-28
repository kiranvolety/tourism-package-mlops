"""Train an XGBoost model on the prepared splits, track with MLflow, and save the best model."""
import os

import joblib
import mlflow
import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import xgboost as xgb

NUMERIC_COLS = [
    "Age", "CityTier", "DurationOfPitch", "NumberOfPersonVisiting",
    "NumberOfFollowups", "PreferredPropertyStar", "NumberOfTrips", "Passport",
    "PitchSatisfactionScore", "OwnCar", "NumberOfChildrenVisiting", "MonthlyIncome",
]
CATEGORICAL_COLS = [
    "TypeofContact", "Occupation", "Gender", "ProductPitched",
    "MaritalStatus", "Designation",
]

MODEL_DIR = "tourism_project/deployment"
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.joblib")


def main():
    Xtrain = pd.read_csv("Xtrain.csv")
    Xtest = pd.read_csv("Xtest.csv")
    ytrain = pd.read_csv("ytrain.csv").iloc[:, 0]
    ytest = pd.read_csv("ytest.csv").iloc[:, 0]

    for col in NUMERIC_COLS:
        Xtrain[col] = Xtrain[col].fillna(Xtrain[col].median())
        Xtest[col] = Xtest[col].fillna(Xtrain[col].median())
    for col in CATEGORICAL_COLS:
        Xtrain[col] = Xtrain[col].fillna(Xtrain[col].mode()[0])
        Xtest[col] = Xtest[col].fillna(Xtrain[col].mode()[0])

    preprocessor = make_column_transformer(
        (StandardScaler(), NUMERIC_COLS),
        (OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS),
    )

    pipeline = make_pipeline(
        preprocessor,
        xgb.XGBClassifier(eval_metric="logloss", random_state=42),
    )

    param_grid = {
        "xgbclassifier__n_estimators": [100, 200],
        "xgbclassifier__max_depth": [3, 5],
        "xgbclassifier__learning_rate": [0.05, 0.1],
    }

    mlflow.set_experiment("tourism-package-prediction")

    with mlflow.start_run():
        search = GridSearchCV(pipeline, param_grid, cv=3, scoring="f1", n_jobs=-1)
        search.fit(Xtrain, ytrain)

        mlflow.log_params(search.best_params_)

        best_model = search.best_estimator_
        preds = best_model.predict(Xtest)
        report = classification_report(ytest, preds, output_dict=True)

        mlflow.log_metric("accuracy", report["accuracy"])
        mlflow.log_metric("f1_score", report["1"]["f1-score"])
        mlflow.log_metric("precision", report["1"]["precision"])
        mlflow.log_metric("recall", report["1"]["recall"])

        print(classification_report(ytest, preds))

        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(best_model, MODEL_PATH)
        mlflow.sklearn.log_model(best_model, "model")

    print(f"Best model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
