#!/usr/bin/env python3
"""
complete_distill_project.py

Complete pipeline using a user-provided Ethanol-Water distillation dataset (or synthetic fallback).
- Loads dataset (looks for common paths)
- Cleans and summarizes data
- Preprocess (scaling + one-hot)
- Train & compare models: Polynomial (Ridge), RandomForest (with light tuning), XGBoost (if available), ANN (if available)
- Diagnostics: parity, residuals, PDP for R and xF, monotonicity checks, error-slice metrics, holdout block (extrapolation)
- Saves models, metrics, figures, and a README in AI_Distillation_Surrogate/
"""

import os, sys, math, warnings, joblib
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import PartialDependenceDisplay
from scipy.optimize import brentq

warnings.filterwarnings("ignore")
np.random.seed(42)

# --------------------------
# Output folders
# --------------------------
ROOT = os.path.abspath("./AI_Distillation_Surrogate")
DATA_DIR = os.path.join(ROOT, "data")
MODELS_DIR = os.path.join(ROOT, "models")
FIG_DIR = os.path.join(ROOT, "figures")
REPORT_DIR = os.path.join(ROOT, "reports")
for d in (DATA_DIR, MODELS_DIR, FIG_DIR, REPORT_DIR):
    os.makedirs(d, exist_ok=True)

# --------------------------
# Locate user's dataset (tries a few common paths)
# --------------------------
candidate_paths = [
    "/mnt/data/distill_out_5k/distill_data_50000_B_input.csv",
    "/mnt/data/distill_out_5k/distill_data_50000_B_input.csv",  # duplicate intentionally
    "./distill_out_5k/distill_data_50000_B_input.csv",
    "./distill_data_50000_B_input.csv",
    "/mnt/data/distill_data_50000_B_input.csv"
]

dataset_path = None
for p in candidate_paths:
    if os.path.exists(p):
        dataset_path = p
        break

# If you uploaded a dataset to another path, set dataset_path variable here (or pass as arg)
if len(sys.argv) > 1:
    # allow overriding by command-line argument
    if os.path.exists(sys.argv[1]):
        dataset_path = sys.argv[1]

# --------------------------
# Synthetic generator (fallback)
# --------------------------
def generate_synthetic(n_samples=50000, random_seed=42):
    np.random.seed(random_seed)
    R_range=(0.8,5.0); B_range=(1.0,6.0); xF_range=(0.2,0.95); F_range=(70,130)
    N_choices=[15,20,25]; q_choices=['Subcooled','Saturated','Superheated']
    k1, kN = 1.2, 0.03
    kq_map = {'Subcooled':0.9, 'Saturated':1.0, 'Superheated':1.05}
    latent_heat, efficiency = 40.0, 0.75

    def separation_score(R,B,N,xF,q):
        return (R/(R+1.0))**0.9 * (B/(B+1.0))**0.7 * (1+kN*(N-15))* (0.5+0.5*xF) * k1 * kq_map[q]
    def xD_from_inputs(R,B,N,xF,q):
        S = separation_score(R,B,N,xF,q)
        return np.clip(xF + (1-xF)*(1-np.exp(-1.8*S)) + np.random.normal(0,0.005),0,1)
    def QR_from_inputs(R,B,N,xF,F,q):
        xD = xD_from_inputs(R,B,N,xF,q)
        lift = max(xD-xF,0)
        return max((latent_heat*F*lift)/efficiency + 10*B + np.random.normal(0,5),0.1)

    rows=[]
    for _ in range(n_samples):
        R=float(np.random.uniform(*R_range))
        B=float(np.random.uniform(*B_range))
        xF=float(np.random.uniform(*xF_range))
        F=float(np.random.uniform(*F_range))
        N=int(np.random.choice(N_choices))
        q=str(np.random.choice(q_choices))
        xD=float(xD_from_inputs(R,B,N,xF,q))
        QR=float(QR_from_inputs(R,B,N,xF,F,q))
        rows.append([R,B,xF,F,N,q,xD,QR])
    df = pd.DataFrame(rows, columns=['R','B','xF','F','N','q','xD','QR'])
    return df
    # --------------------------
# Load or generate dataset
# --------------------------
if dataset_path is None:
    print("No dataset found in default locations — generating synthetic 50k dataset as fallback.")
    df = generate_synthetic(50000)
else:
    print("Loading dataset from:", dataset_path)
    df = pd.read_csv(dataset_path)

# Save a copy to project folder
df.to_csv(os.path.join(DATA_DIR, "distill_data_used.csv"), index=False)

# --------------------------
# Quick EDA & cleaning
# --------------------------
print("\nDataset preview:")
print(df.head())
print("\nShape:", df.shape)
print("\nColumns:", df.columns.tolist())

# Ensure required columns exist
required_cols = {'R','B','xF','F','N','q','xD','QR'}
if not required_cols.issubset(set(df.columns)):
    raise ValueError(f"Dataset must contain columns: {required_cols}")

# Convert types if needed
df['q'] = df['q'].astype(str)
df['N'] = df['N'].astype(int)

# Remove rows with NaN or invalid xD/QR
n0 = df.shape[0]
df = df.dropna(subset=list(required_cols))
# clip xD to [0,1] and drop negative QR
df = df[df['xD'].between(0,1)]
df = df[df['QR'] >= 0]
print(f"Dropped {n0 - df.shape[0]} bad rows; remaining {df.shape[0]} rows.")

# Basic stats
print("\nBasic statistics (inputs):")
print(df[['R','B','xF','F']].describe().T)

# --------------------------
# Preprocessing & splits
# --------------------------
feature_cols = ['R','B','xF','F','N','q']
target_cols = ['xD','QR']

# Create a block holdout (extrapolation) — hold out an R band (example: 3.5 - 4.5)
R_hold_min, R_hold_max = 3.5, 4.5
mask_hold = (df['R'] >= R_hold_min) & (df['R'] <= R_hold_max)
df_hold = df[mask_hold].copy()
df_rest = df[~mask_hold].copy()

# Train/validation/test split from remaining data
train_df, test_df = train_test_split(df_rest, test_size=0.20, random_state=42)
print("\nTrain / Test / Hold sizes:", train_df.shape[0], test_df.shape[0], df_hold.shape[0])

# Preprocessor: numeric scale + one-hot
num_cols = ['R','B','xF','F']
cat_cols = ['N','q']
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
])

# Fit preprocessor on train
preprocessor.fit(train_df[feature_cols])

def prepare_arrays(df_in):
    X = preprocessor.transform(df_in[feature_cols])
    y = df_in[target_cols].values
    return X, y

X_train, y_train = prepare_arrays(train_df)
X_test, y_test = prepare_arrays(test_df)
X_hold, y_hold = prepare_arrays(df_hold) if df_hold.shape[0] > 0 else (None, None)

# For pipelines that accept DataFrames, keep original feature DataFrames handy
# (we'll also create small wrappers where required)
# --------------------------
# Models: train & compare
# --------------------------
results = []

# Helper metrics
def metrics(y_true, y_pred):
    return {
        'MAE_xD': mean_absolute_error(y_true[:,0], y_pred[:,0]),
        'RMSE_xD': math.sqrt(mean_squared_error(y_true[:,0], y_pred[:,0])),
        'R2_xD': r2_score(y_true[:,0], y_pred[:,0]),
        'MAE_QR': mean_absolute_error(y_true[:,1], y_pred[:,1]),
        'RMSE_QR': math.sqrt(mean_squared_error(y_true[:,1], y_pred[:,1])),
        'R2_QR': r2_score(y_true[:,1], y_pred[:,1])
    }

# Model store (name -> pipeline / wrapper)
model_store = {}

# ---- Baseline: Polynomial Ridge (degree 2) ----
print("\nTraining Polynomial Ridge (degree=2) ...")
poly_pipe = Pipeline([
    ('pre', preprocessor),
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('lr', Ridge(alpha=1.0))
])
poly_pipe.fit(train_df[feature_cols], train_df[target_cols])
y_pred = poly_pipe.predict(test_df[feature_cols])
m_poly = metrics(y_test, y_pred)
m_poly['model'] = 'Polynomial_Ridge'
results.append(m_poly)
model_store['Polynomial_Ridge'] = poly_pipe
print("Polynomial done. R2_xD=%.4f, R2_QR=%.4f" % (m_poly['R2_xD'], m_poly['R2_QR']))

# ---- RandomForest (with light RandomizedSearchCV tuning) ----
print("\nTraining RandomForest (with light tuning)...")
rf_pipe = Pipeline([
    ('pre', preprocessor),
    ('rf', RandomForestRegressor(random_state=42, n_jobs=-1))
])
param_dist = {
    'rf__n_estimators': [100, 200, 300, 400, 500],
    'rf__max_depth': [10, 15, 20, 25, None],
    'rf__min_samples_split': [2, 5, 10],
    'rf__min_samples_leaf': [1, 2, 4],
    'rf__max_features': ['auto', 'sqrt', 'log2']
}
r_search = RandomizedSearchCV(rf_pipe, param_distributions=param_dist, n_iter=20, cv=5,
                              scoring='r2', verbose=2, n_jobs=-1, random_state=42)
r_search.fit(train_df[feature_cols], train_df[target_cols])
best_rf = r_search.best_estimator_
model_store['RandomForest'] = best_rf
y_pred = best_rf.predict(test_df[feature_cols])
m_rf = metrics(y_test, y_pred); m_rf['model']='RandomForest'
results.append(m_rf)
print("RandomForest done. Best params:", r_search.best_params_)
print("R2_xD=%.4f, R2_QR=%.4f" % (m_rf['R2_xD'], m_rf['R2_QR']))

# ---- XGBoost (optional) ----
have_xgb = False
try:
    import xgboost as xgb
    print("\nTraining XGBoost (light) ...")
    xgb_pipe = Pipeline([
        ('pre', preprocessor),
        ('xgb', xgb.XGBRegressor(n_estimators=300, random_state=42, objective='reg:squarederror', n_jobs=-1))
    ])
    xgb_pipe.fit(train_df[feature_cols], train_df[target_cols])
    model_store['XGBoost'] = xgb_pipe
    y_pred = xgb_pipe.predict(test_df[feature_cols])
    m_xgb = metrics(y_test, y_pred); m_xgb['model']='XGBoost'
    results.append(m_xgb)
    have_xgb = True
    print("XGBoost done. R2_xD=%.4f, R2_QR=%.4f" % (m_xgb['R2_xD'], m_xgb['R2_QR']))
except Exception as e:
    print("XGBoost skipped (not available or failed):", e)

# ---- ANN (TensorFlow Keras small MLP) (optional) ----
have_ann = False
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping

    # Wrap for prediction by DataFrame: create simple wrapper
    class ANNWrapper:
        def __init__(self, preproc, model):
            self.preproc = preproc
            self.model = model
        def predict(self, X_df):
            Xp = self.preproc.transform(X_df)
            return self.model.predict(Xp)

    print("\nTraining small ANN ...")
    # Build numeric training arrays (preprocessor already fit)
    X_train_pre = preprocessor.transform(train_df[feature_cols])
    X_val_pre = preprocessor.transform(test_df[feature_cols])

    input_dim = X_train_pre.shape[1]
    def build_ann_model():
       model = Sequential()
       model.add(Dense(128, input_dim=input_dim, activation='relu'))
       model.add(Dropout(0.2))
       model.add(Dense(64, activation='relu'))
       model.add(Dropout(0.2))
       model.add(Dense(32, activation='relu'))
       model.add(Dense(2, activation='linear'))  # Predict xD and QR simultaneously
       model.compile(optimizer='adam', loss='mse', metrics=['mae'])
       return model

    ann = build_ann_model()
    es = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    ann.fit(X_train_pre, train_df[target_cols].values, validation_split=0.2,
            epochs=200, batch_size=64, callbacks=[es], verbose=2)
    # Wrap for prediction by DataFrame: create simple wrapper

    ann_wrap = ANNWrapper(preprocessor, ann)
    model_store['ANN'] = ann_wrap
    y_pred = ann_wrap.predict(test_df[feature_cols])
    m_ann = metrics(y_test, y_pred); m_ann['model']='ANN'
    results.append(m_ann)
    have_ann = True
    print("ANN done. R2_xD=%.4f, R2_QR=%.4f" % (m_ann['R2_xD'], m_ann['R2_QR']))
except Exception as e:
    print("ANN skipped (TensorFlow not installed or failed):", e)

# --------------------------
# Save model metrics summary
# --------------------------
res_df = pd.DataFrame(results)
res_df.to_csv(os.path.join(REPORT_DIR, "model_comparison_summary.csv"), index=False)
print("\nModel comparison table saved to:", os.path.join(REPORT_DIR, "model_comparison_summary.csv"))
print(res_df[['model','R2_xD','R2_QR']])

# --------------------------
# Select best model (highest avg R2)
# --------------------------
res_df['avg_R2'] = (res_df['R2_xD'] + res_df['R2_QR'])/2.0
res_df = res_df.sort_values('avg_R2', ascending=False).reset_index(drop=True)
best_name = res_df.loc[0,'model']
print("\nBest model chosen:", best_name)
best_model = model_store.get(best_name)
# Save best model
# If ANN wrapper (not picklable by joblib easily), save Keras weights and preprocessor separately
if best_name == 'ANN' and have_ann:
    # Save keras model
    ann.save(os.path.join(MODELS_DIR,'ann_model.h5'))
    joblib.dump(preprocessor, os.path.join(MODELS_DIR, 'preprocessor_for_ann.pkl'))
    print("Saved ANN (h5) and preprocessor.")
else:
    joblib.dump(best_model, os.path.join(MODELS_DIR, 'best_model.pkl'))
    print("Saved best model pipeline to:", os.path.join(MODELS_DIR, 'best_model.pkl'))

# --------------------------
# Diagnostics and Figures
# --------------------------
def save_parity_and_residuals(model, X_df, y_true, tag):
    # model: pipeline-like with .predict accepting DataFrame OR numpy arrays
    try:
        # Try predicting directly with DataFrame if the model/pipeline supports it
        if hasattr(model, 'predict'):
             y_pred = model.predict(X_df)
        else:
             # Fallback for models that expect transformed arrays (like raw Keras model)
             Xp = preprocessor.transform(X_df)
             y_pred = model.predict(Xp)
    except Exception:
        # If direct prediction with DataFrame fails, try transforming first
        Xp = preprocessor.transform(X_df)
        y_pred = model.predict(Xp)

    # Parity
    fig, axes = plt.subplots(1,2,figsize=(10,4))
    for i,col in enumerate(['xD','QR']):
        axes[i].scatter(y_true[:,i], y_pred[:,i], s=8, alpha=0.4)
        mn = min(y_true[:,i].min(), y_pred[:,i].min())
        mx = max(y_true[:,i].max(), y_pred[:,i].max())
        axes[i].plot([mn,mx],[mn,mx], 'k--')
        axes[i].set_xlabel('True'); axes[i].set_ylabel('Pred')
        axes[i].set_title(f'{tag} parity: {col}')
    plt.tight_layout()
    path = os.path.join(FIG_DIR, f'parity_{tag}.png')
    plt.savefig(path, dpi=150); plt.close()
    # Residuals
    residuals = y_true - y_pred
    fig, axes = plt.subplots(1,2,figsize=(10,4))
    axes[0].scatter(y_pred[:,0], residuals[:,0], s=8, alpha=0.4); axes[0].axhline(0, color='k', linestyle='--')
    axes[0].set_xlabel('Pred xD'); axes[0].set_ylabel('Residual xD'); axes[0].set_title(f'Residuals xD - {tag}')
    axes[1].scatter(y_pred[:,1], residuals[:,1], s=8, alpha=0.4); axes[1].axhline(0, color='k', linestyle='--')
    axes[1].set_xlabel('Pred QR'); axes[1].set_ylabel('Residual QR'); axes[1].set_title(f'Residuals QR - {tag}')
    plt.tight_layout()
    rpath = os.path.join(FIG_DIR, f'residuals_{tag}.png')
    plt.savefig(rpath, dpi=150); plt.close()
    return y_pred

# Parity/residuals on test set
print("\nCreating parity and residual plots for best model on test set...")
# Use DataFrame of features for test_df
y_pred_test = save_parity_and_residuals(best_model, test_df[feature_cols], y_test, 'test_best')

# Parity/residuals on holdout block (extrapolation)
if df_hold.shape[0] > 0:
    print("Creating parity/residuals on held-out R-block...")
    y_pred_hold = save_parity_and_residuals(best_model, df_hold[feature_cols], y_hold, 'holdout_Rblock')
    # Save holdout metrics
    hold_metrics = metrics(y_hold, y_pred_hold)
    pd.DataFrame([hold_metrics], index=['hold_R_block']).to_csv(os.path.join(REPORT_DIR,'holdout_block_metrics.csv'))
    print("Holdout block metrics saved.")
else:
    print("No holdout R-block to analyze.")

# Partial dependence plots for R and xF (use best_model if compatible)
try:
    print("Attempting Partial Dependence plots for R and xF...")
    # PartialDependenceDisplay requires an estimator that accepts DataFrame and pipeline is OK.
    feat_names = feature_cols
    # For convenience, compute PDP on the test_df DataFrame
    features_to_plot = ['R','xF']
    fig, axes = plt.subplots(1,2,figsize=(10,4))
    PartialDependenceDisplay.from_estimator(best_model, test_df[feature_cols], features_to_plot, ax=axes)
    ppath = os.path.join(FIG_DIR, 'pdp_R_xF.png')
    plt.tight_layout(); plt.savefig(ppath, dpi=150); plt.close()
    print("Saved PDP to", ppath)
except Exception as e:
    print("PDP failed/skipped:", e)
    # Monotonicity checks: vary R while keeping other inputs fixed (several base cases)
def monotonicity_check(model, base_row, var='R', vals=np.linspace(0.8,5.0,40)):
    rows=[]
    for v in vals:
        r = base_row.copy()
        r[var] = v
        rows.append(r)
    Xs = pd.DataFrame(rows)[feature_cols]
    try:
        preds = model.predict(Xs)
    except Exception:
        preds = model.predict(preprocessor.transform(Xs))
    xD_preds = preds[:,0]
    diffs = np.diff(xD_preds)
    violations = np.sum(diffs < -1e-4)
    # Save plot
    plt.figure(figsize=(5,3))
    plt.plot(vals, xD_preds, marker='.', markersize=4)
    plt.xlabel(var); plt.ylabel('xD'); plt.title(f"Monotonicity w.r.t {var} (violations={violations})")
    plt.grid(True)
    plt.tight_layout()
    return violations, vals, xD_preds

print("\nRunning monotonicity checks (3 base cases) ...")
base_examples = [
    {'R':2.0,'B':2.5,'xF':0.5,'F':100,'N':20,'q':'Saturated'},
    {'R':2.0,'B':3.0,'xF':0.3,'F':90,'N':25,'q':'Subcooled'},
    {'R':2.0,'B':1.5,'xF':0.8,'F':110,'N':15,'q':'Superheated'}
]
mon_results = []
for i, base in enumerate(base_examples):
    v, vals, preds = monotonicity_check(best_model, base)
    mon_results.append({'case': i, 'violations': int(v)})
    plt.savefig(os.path.join(FIG_DIR, f"monotonicity_case_{i}.png")); plt.close()
pd.DataFrame(mon_results).to_csv(os.path.join(REPORT_DIR, "monotonicity_summary.csv"), index=False)
print("Monotonicity summary saved.")

# Error-slice metrics: high purity xD >= 0.95, low feed xF <= 0.3
print("\nComputing error-slice metrics...")
def compute_slice_metrics(df_slice):
    if df_slice.shape[0] == 0:
        return {}
    Xs = df_slice[feature_cols]
    try:
        ypred = best_model.predict(Xs)
    except Exception:
        ypred = best_model.predict(preprocessor.transform(Xs))
    return metrics(df_slice[target_cols].values, ypred)

slice_high_purity = test_df[test_df['xD'] >= 0.95]
slice_low_feed = test_df[test_df['xF'] <= 0.3]
slice_metrics = {
    'high_purity_xD>=0.95': compute_slice_metrics(slice_high_purity),
    'low_feed_xF<=0.3': compute_slice_metrics(slice_low_feed)
}
pd.DataFrame(slice_metrics).T.to_csv(os.path.join(REPORT_DIR, "error_slices_metrics.csv"))
print("Error-slice metrics saved.")

# Optional: SHAP explainability for best tree model (if shap available and model is tree-based)
try:
    import shap
    print("\nAttempting SHAP explanation (for tree models)...")
    if best_name in ['RandomForest','XGBoost']:
        # take a small sample for SHAP (speed)
        sample_df = test_df[feature_cols].sample(n=min(200, test_df.shape[0]), random_state=42)
        # transform to arrays for tree explainer if needed
        try:
            explainer = shap.Explainer(best_model.named_steps.get('rf', best_model.named_steps.get('xgb')))
            shap_vals = explainer(best_model.named_steps['pre'].transform(sample_df))
            shap.summary_plot(shap_vals, features=sample_df, show=False)
            plt.title("SHAP summary")
            plt.savefig(os.path.join(FIG_DIR, 'shap_summary.png'), dpi=150); plt.close()
            print("Saved shap_summary.png")
        except Exception as e:
            print("SHAP tree explainer failed:", e)
    else:
        print("Best model not tree-based; skipping SHAP.")
except Exception as e:
    print("SHAP not available or failed:", e)

# --------------------------
# Save outputs & README
# --------------------------
# Save preprocessor
joblib.dump(preprocessor, os.path.join(MODELS_DIR, 'preprocessor.pkl'))

# Save copy of used dataset
df.to_csv(os.path.join(DATA_DIR, 'distill_data_used.csv'), index=False)

# Save results summary
res_df.to_csv(os.path.join(REPORT_DIR, 'model_ranking.csv'), index=False)

# Simple README
with open(os.path.join(ROOT, 'README.txt'),'w') as f:
    f.write("AI Distillation Surrogate Project\n")
    f.write("Folder structure:\n")
    f.write(" data/: dataset in use\n")
    f.write(" models/: saved models and preprocessor\n")
    f.write(" figures/: parity, residuals, PDP, monotonicity, etc.\n")
    f.write(" reports/: model metrics, holdout/extrapolation metrics, error slices\n\n")
    f.write("How to reproduce: run complete_distill_project.py with Python environment containing numpy,pandas,scikit-learn,matplotlib,joblib. Optional: xgboost, tensorflow (for ANN), shap (for explainability).\n")

print("\nAll done. Outputs saved under:", ROOT)
print("Key files:")
print(" - Data copy: ", os.path.join(DATA_DIR, 'distill_data_used.csv'))
print(" - Best model: ", os.path.join(MODELS_DIR, 'best_model.pkl') if best_name != 'ANN' else os.path.join(MODELS_DIR, 'ann_model.h5'))
print(" - Metrics & plots: ", REPORT_DIR, FIG_DIR)