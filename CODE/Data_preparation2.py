# Data_preprocessing.py
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def build_preprocessor():
    numeric = ["R","B","xF","F"]
    categorical = ["N","q"]
    preproc = ColumnTransformer([
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(sparse=False, handle_unknown="ignore"), categorical)
    ])
    return preproc

