# Train_models.py
import pandas as pd, joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import math, os

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def build_preprocessor():
    numeric = ["R","B","xF","F"]
    categorical = ["N","q"]
    print("Preprocessor created for numeric + categorical features") # Changed from logging.info to print for Colab output
    return ColumnTransformer([
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical)
    ])


def evaluate(y_true, y_pred):
    return {
        "MAE_xD": mean_absolute_error(y_true["xD"], y_pred[:,0]),
        "RMSE_xD": math.sqrt(mean_squared_error(y_true["xD"], y_pred[:,0])),
        "R2_xD": r2_score(y_true["xD"], y_pred[:,0]),
        "MAE_QR": mean_absolute_error(y_true["QR"], y_pred[:,1]),
        "RMSE_QR": math.sqrt(mean_squared_error(y_true["QR"], y_pred[:,1])),
        "R2_QR": r2_score(y_true["QR"], y_pred[:,1]),
    }

def train_models(data_csv="data/distill_data.csv"):
    df = pd.read_csv(data_csv)
    X = df[["R","B","xF","F","N","q"]]
    y = df[["xD","QR"]]
    Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.2,random_state=42)

    preproc = build_preprocessor()
    results = {}
    models = {}

    # Polynomial Ridge
    poly = Pipeline([("pre", preproc),
                     ("poly", PolynomialFeatures(2, include_bias=False)),
                     ("ridge", Ridge(alpha=1.0))])
    poly.fit(Xtr,ytr)
    results["PolynomialRidge"] = evaluate(yte, poly.predict(Xte))
    models["PolynomialRidge"] = poly

    # RandomForest
    rf = Pipeline([("pre", preproc),
                   ("rf", RandomForestRegressor(n_estimators=200, max_depth=18, random_state=42, n_jobs=-1))])
    rf.fit(Xtr,ytr)
    results["RandomForest"] = evaluate(yte, rf.predict(Xte))
    models["RandomForest"] = rf

    os.makedirs("models", exist_ok=True)
    for name, model in models.items():
        joblib.dump(model, f"models/{name}.pkl")

    os.makedirs("reports", exist_ok=True) # Add this line
    pd.DataFrame(results).T.to_csv("reports/metrics.csv")
    print("✅ Models trained, metrics saved to reports/metrics.csv")
    return models, results

if __name__ == "__main__":
    train_models()