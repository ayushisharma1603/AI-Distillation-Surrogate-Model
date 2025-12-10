# Diagnostics.py
import matplotlib.pyplot as plt, pandas as pd, os
from sklearn.inspection import PartialDependenceDisplay
import joblib

def plot_parity(model, Xte, yte, name="RandomForest"):
    y_pred = model.predict(Xte)
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.scatter(yte["xD"], y_pred[:,0], s=5, alpha=0.5)
    plt.plot([0,1],[0,1],'k--'); plt.xlabel("True xD"); plt.ylabel("Pred xD"); plt.title(f"{name} - xD")
    plt.subplot(1,2,2)
    mn,mx=min(yte["QR"].min(),y_pred[:,1].min()), max(yte["QR"].max(),y_pred[:,1].max())
    plt.scatter(yte["QR"], y_pred[:,1], s=5, alpha=0.5)
    plt.plot([mn,mx],[mn,mx],'k--'); plt.xlabel("True QR"); plt.ylabel("Pred QR"); plt.title(f"{name} - QR")
    os.makedirs("figures", exist_ok=True)
    plt.tight_layout(); plt.savefig(f"figures/parity_{name}.png", dpi=150)
    plt.close()

def diagnostics():
    df = pd.read_csv("data/distill_data.csv")
    X = df[["R","B","xF","F","N","q"]]
    y = df[["xD","QR"]]
    rf = joblib.load("models/RandomForest.pkl")

    from sklearn.model_selection import train_test_split
    Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.2,random_state=42)

    plot_parity(rf, Xte, yte)

    # PDP for xD
    fig_xD, ax_xD = plt.subplots(1, 2, figsize=(10, 4))
    PartialDependenceDisplay.from_estimator(rf, Xte, ["R", "xF"], target=0, ax=ax_xD)
    fig_xD.suptitle("Partial Dependence Plots for xD", fontsize=12)
    plt.tight_layout(); plt.savefig("figures/pdp_RF_xD.png", dpi=150); plt.close(fig_xD)

    # PDP for QR
    fig_QR, ax_QR = plt.subplots(1, 2, figsize=(10, 4))
    PartialDependenceDisplay.from_estimator(rf, Xte, ["R", "xF"], target=1, ax=ax_QR)
    fig_QR.suptitle("Partial Dependence Plots for QR", fontsize=12)
    plt.tight_layout(); plt.savefig("figures/pdp_RF_QR.png", dpi=150); plt.close(fig_QR)


if __name__ == "__main__":
    diagnostics()