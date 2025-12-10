AI Distillation Surrogate Project
================================

Validation Checklist (Quick Run)
--------------------------------
Run:   python complete_distill_project.py
Check: reports/model_ranking.csv  → best model ranking
       figures/parity_test_best.png → points close to diagonal = good fit

Project overview
----------------
This project builds AI/ML surrogate models to predict distillate purity (xD) and reboiler duty (QR)
for an Ethanol–Water binary distillation column operating at 1 atm. Surrogates (Random Forest, XGBoost,
Polynomial Ridge, and optional ANN) are trained on a dataset of 50,000 samples generated from DWSIM
or via the synthetic generator. The surrogate enables fast prediction and can be used for optimization
(e.g., minimize QR while maintaining xD ≥ 0.95).

Folder structure
----------------
AI_Distillation_Surrogate/
│
├── data/
│   └── distill_data_used.csv        # Cleaned dataset used for training and evaluation (50,000 samples)
│
├── models/
│   ├── best_model.pkl               # Saved best model pipeline (or ann_model.h5 if ANN chosen)
│   ├── preprocessor.pkl             # Preprocessing pipeline (StandardScaler + OneHotEncoder)
│   └── preprocessor_for_ann.pkl     # (if ANN used / saved separately)
│
├── figures/
│   ├── parity_test_best.png         # Parity plot: true vs predicted (test set) for xD and QR
│   ├── residuals_test_best.png      # Residual plots (test set)
│   ├── pdp_R_xF.png                 # Partial dependence for R and xF on xD
│   ├── monotonicity_case_0.png      # Monotonicity test case 0 (xD vs R)
│   ├── monotonicity_case_1.png      # Monotonicity test case 1 (xD vs R)
│   ├── monotonicity_case_2.png      # Monotonicity test case 2 (xD vs R)
│   └── shap_summary.png             # (Optional) SHAP summary for tree models
│
├── reports/
│   ├── model_comparison_summary.csv # Metrics (MAE, RMSE, R2) for each model
│   ├── model_ranking.csv            # Models ranked by avg R² across xD and QR
│   ├── holdout_block_metrics.csv    # Performance on R holdout block (R ∈ [3.5,4.5])
│   ├── monotonicity_summary.csv     # Monotonicity violations count
│   └── error_slices_metrics.csv     # Metrics on high-purity and low-feed slices
│
├── complete_distill_project.py      # Main pipeline script — data -> preprocess -> train -> diagnostics
└── README.txt                       # This file

Requirements
------------
For reproducibility, install exact package versions.  
Generate with:
   pip freeze > requirements.txt

How to run:
1. Install dependencies: pip install numpy pandas scikit-learn matplotlib joblib openpyxl
2. To inspect dataset: python -c "import pandas as pd; print(pd.read_csv('AI_Distillation_Surrogate/data/distill_data.csv').head())"
3. To run training using your final code, place it inside code/ and run as instructed in your script.

Example requirements.txt:
numpy==1.26.4
pandas==2.2.2
scikit-learn==1.5.2
matplotlib==3.9.2
joblib==1.4.2
xgboost==2.1.1
tensorflow==2.17.0
shap==0.46.0

Quick usage example
-------------------
import joblib, pandas as pd

# Load saved best model
model = joblib.load('models/best_model.pkl')

# Define one sample input
sample = pd.DataFrame([{
    'R': 2.5,
    'B': 3.0,
    'xF': 0.6,
    'F': 100,
    'N': 20,
    'q': 'Saturated'
}])

# Predict distillate purity (xD) and reboiler duty (QR)
print(model.predict(sample))

Expected output format:
[[0.9523, 2540.7]]
# → xD ≈ 0.95 (95% purity), QR ≈ 2540 energy units

Sample results (from reports/model_comparison_summary.csv)
----------------------------------------------------------
| Model             | R²_xD | R²_QR | MAE_xD  | MAE_QR |
|-------------------|-------|-------|---------|--------|
| Polynomial_Ridge  | 0.87  | 0.82  | 0.015   | 210.5  |
| RandomForest      | 0.96  | 0.95  | 0.007   | 95.2   |
| XGBoost           | 0.97  | 0.96  | 0.006   | 88.7   |
| ANN               | 0.95  | 0.94  | 0.008   | 102.3  |

(Values are illustrative — your actual metrics are in reports/model_comparison_summary.csv.)

Optimization Example
--------------------
The surrogate can be used to optimize operating conditions.

Optimization goal:
- Minimize QR
- Subject to xD ≥ 0.95
- Variables optimized: Reflux ratio (R: 0.8–5.0), Boilup ratio (B: 1.0–6.0)

The optimization script (inside complete_distill_project.py) uses Scipy SLSQP with a grid-search fallback.
Results are saved in reports/optimization_result.json.

Example snippet to read optimization output:

import json
with open("reports/optimization_result.json") as f:
    opt = json.load(f)

print("Optimized result:", opt)

Sample output:
{'method': 'SLSQP', 'R': 2.91, 'B': 3.42, 'xD': 0.956, 'QR': 2480.5}


Notes on inputs and script behavior
----------------------------------
- The script expects the dataset to contain columns: R, B, xF, F, N, q, xD, QR.
- If no dataset is found, the script generates a synthetic dataset (configured inside the script).
- The script creates a block holdout for extrapolation testing: R ∈ [3.5, 4.5]. Edit R_hold_min / R_hold_max in the script to change this.
- Preprocessing: numeric features are StandardScaled, categorical features (N, q) are one-hot encoded.
- Models: Polynomial (Ridge) baseline, RandomForest (tuned via RandomizedSearchCV), XGBoost (if installed), ANN (Keras/TensorFlow, if installed).
- Outputs: models saved under /models/, plots under /figures/, metric CSVs under /reports/.

Interpreting key outputs
------------------------
- model_comparison_summary.csv: rows = models; columns include MAE_xD, RMSE_xD, R2_xD, MAE_QR, RMSE_QR, R2_QR.
- parity_test_best.png: points close to diagonal => model predicts well; systematic bias visible as off-diagonal shift.
- residuals_test_best.png: residuals should be symmetrically distributed around zero; patterns indicate bias.
- pdp_R_xF.png: shows marginal effect of R and xF on predicted xD, helpful to check physical behavior.
- monotonicity_summary.csv: low number of violations is preferred — indicates model respects physical monotonicity with R.
- error_slices_metrics.csv: performance in critical regions, e.g., xD >= 0.95 (high-purity).

Reproducibility & environment
-----------------------------
- Record exact python packages:
   pip freeze > requirements.txt
- For reproducibility, set random seed inside the script (already set to 42).
- Large models (RF/XGBoost/ANN) may require significant memory; run on a machine with sufficient RAM.

Troubleshooting
---------------
- "ModuleNotFoundError: xgboost" — install xgboost or run the script without it (XGBoost is optional).
- TensorFlow errors on CPU-only machines — install `tensorflow-cpu` or skip ANN training.
- If figures are not generated, check the /figures/ folder and stderr output for exceptions.

Known limitations & future work
------------------------------
- Dataset is synthetic if DWSIM data not provided; while synthetic is useful for method development, final submission should include at least some DWSIM-generated data.
- Consider physics-informed losses or monotonicity constraints to improve physical consistency.
- Extend to multi-component mixtures, pressures ≠ 1 atm, or include additional operating variables.

Contact & authorship
--------------------
Author: <Ayushi Sharma>
Email: <ayushi.23bai10569@vitbhopal.ac.in>



