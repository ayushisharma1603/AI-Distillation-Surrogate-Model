<p align="center">
  <img width="100%" src="https://capsule-render.vercel.app/api?type=wave&color=0:6a11cb,100:2575fc&height=180&section=header&text=AI%20Distillation%20Surrogate%20Project&fontSize=35&fontColor=ffffff" />
</p>

<p align="center">
  <img src="https://img.shields.io/badge/ML-Surrogate%20Model-orange?style=for-the-badge">
  <img src="https://img.shields.io/badge/Distillation-Column%20Modeling-blue?style=for-the-badge">
  <img src="https://img.shields.io/badge/Python-3.10+-yellow?style=for-the-badge&logo=python">
</p>
---

# 📘 AI Distillation Surrogate Project

This project builds **Machine Learning surrogate models** to predict:

- **Distillate purity (xD)**
- **Reboiler duty (QR)**  

for a **binary Ethanol–Water distillation column operating at 1 atm**.  
The surrogate provides **fast, accurate predictions** and enables **optimization** of operating conditions.

---

## 🚀 Features

- ML models: **Polynomial Ridge**, **Random Forest**, **XGBoost**, **ANN**
- Dataset: **50,000 samples** (DWSIM or synthetic generator)
- Auto-generated outputs:
  - Best model (`/models/best_model.pkl`)
  - Metrics & comparisons (`/reports`)
  - Parity plots, residuals, PDPs, monotonicity (`/figures`)
- Optimization using **SLSQP**  
  → Minimize **QR** with constraint **xD ≥ 0.95**

---

## 📁 Project Structure
data/ # Dataset
models/ # Saved model files + preprocessors
figures/ # Diagnostic plots
reports/ # Performance metrics
complete_distill_project.py # Full pipeline script

