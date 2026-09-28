
"""Train an XGBoost model on the prepared splits, track with MLflow, and save the best model."""    main()

import osif __name__ == "__main__":



import joblib

import mlflow    print(f"Best model saved to {MODEL_PATH}")

import pandas as pd

from sklearn.compose import make_column_transformer        mlflow.sklearn.log_model(best_model, "model")

from sklearn.metrics import classification_report        joblib.dump(best_model, MODEL_PATH)

from sklearn.model_selection import GridSearchCV        os.makedirs(MODEL_DIR, exist_ok=True)

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import OneHotEncoder, StandardScaler        print(classification_report(ytest, preds))

import xgboost as xgb

        mlflow.log_metric("recall", report["1"]["recall"])

NUMERIC_COLS = [        mlflow.log_metric("precision", report["1"]["precision"])

    "Age", "CityTier", "DurationOfPitch", "NumberOfPersonVisiting",        mlflow.log_metric("f1_score", report["1"]["f1-score"])

    "NumberOfFollowups", "PreferredPropertyStar", "NumberOfTrips", "Passport",        mlflow.log_metric("accuracy", report["accuracy"])

    "PitchSatisfactionScore", "OwnCar", "NumberOfChildrenVisiting", "MonthlyIncome",

]        report = classification_report(ytest, preds, output_dict=True)

CATEGORICAL_COLS = [        preds = best_model.predict(Xtest)

    "TypeofContact", "Occupation", "Gender", "ProductPitched",        best_model = search.best_estimator_

    "MaritalStatus", "Designation",

]        mlflow.log_params(search.best_params_)



MODEL_DIR = "tourism_project/deployment"        search.fit(Xtrain, ytrain)

MODEL_PATH = os.path.join(MODEL_DIR, "best_model.joblib")        search = GridSearchCV(pipeline, param_grid, cv=3, scoring="f1", n_jobs=-1)

    with mlflow.start_run():



def main():    mlflow.set_experiment("tourism-package-prediction")

    Xtrain = pd.read_csv("Xtrain.csv")

    Xtest = pd.read_csv("Xtest.csv")    }

    ytrain = pd.read_csv("ytrain.csv").iloc[:, 0]        "xgbclassifier__learning_rate": [0.05, 0.1],

    ytest = pd.read_csv("ytest.csv").iloc[:, 0]        "xgbclassifier__max_depth": [3, 5],

        "xgbclassifier__n_estimators": [100, 200],

    for col in NUMERIC_COLS:    param_grid = {

        Xtrain[col] = Xtrain[col].fillna(Xtrain[col].median())

        Xtest[col] = Xtest[col].fillna(Xtrain[col].median())    )

    for col in CATEGORICAL_COLS:        xgb.XGBClassifier(eval_metric="logloss", random_state=42),

        Xtrain[col] = Xtrain[col].fillna(Xtrain[col].mode()[0])        preprocessor,

        Xtest[col] = Xtest[col].fillna(Xtrain[col].mode()[0])    pipeline = make_pipeline(



    preprocessor = make_column_transformer(    )

        (StandardScaler(), NUMERIC_COLS),        (OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS),
