import pandas as pd, joblib, logging, os
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def build_preprocessor():
    numeric = ["R","B","xF","F"]
    categorical = ["N","q"]
    logging.info("Preprocessor created for numeric + categorical features")
    return ColumnTransformer([
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical)
    ])


def optimize_rf(data_csv="data/distill_data.csv"):
    df = pd.read_csv(data_csv)
    X,y = df[["R","B","xF","F","N","q"]], df[["xD","QR"]]
    Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.2,random_state=42)

    preproc = build_preprocessor()
    rf = RandomForestRegressor(random_state=42, n_jobs=-1)

    param_grid = {
        "n_estimators": [100,200,300],
        "max_depth": [10,15,20],
        "min_samples_split": [2,5,10]
    }

    search = RandomizedSearchCV(rf, param_grid, n_iter=5, cv=3, scoring="r2", n_jobs=-1, verbose=1)
    pipe = Pipeline([("pre", preproc), ("rf", search)])
    pipe.fit(Xtr,ytr)

    os.makedirs("models", exist_ok=True)
    joblib.dump(pipe, "models/RandomForest_Optimized.pkl")
    logging.info("Optimized RandomForest saved → models/RandomForest_Optimized.pkl")
    return pipe

if __name__ == "__main__":
    optimize_rf()