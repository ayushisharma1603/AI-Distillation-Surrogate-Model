# Data_preparation.py
import numpy as np, pandas as pd, os

def generate_dataset(n_samples=50000, seed=42, out_csv="data/distill_data.csv"):
    np.random.seed(seed)

    R = np.random.uniform(0.8,5.0,n_samples)
    B = np.random.uniform(1.0,6.0,n_samples)
    xF = np.random.uniform(0.2,0.95,n_samples)
    F = np.random.uniform(70,130,n_samples)
    N = np.random.choice([15,20,25], n_samples)
    q = np.random.choice(["Subcooled","Saturated","Superheated"], n_samples)

    k1, kN = 1.2, 0.03
    kq_map = {"Subcooled":0.9,"Saturated":1.0,"Superheated":1.05}
    latent_heat, efficiency = 40.0, 0.75

    def separation_score(R,B,N,xF,q):
        return (R/(R+1))**0.9 * (B/(B+1))**0.7 * (1+kN*(N-15)) * (0.5+0.5*xF) * k1 * np.array([kq_map[qi] for qi in q])

    S = separation_score(R,B,N,xF,q)
    xD = np.clip(xF + (1-xF)*(1-np.exp(-1.8*S)) + np.random.normal(0,0.005,n_samples), 0,1)
    lift = np.maximum(xD-xF,0.0)
    QR = (latent_heat*F*lift)/efficiency + 10*B + np.random.normal(0,5,n_samples)
    QR = np.clip(QR, 0.1, None)

    df = pd.DataFrame({"R":R,"B":B,"xF":xF,"F":F,"N":N,"q":q,"xD":xD,"QR":QR})
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"✅ Dataset saved to {out_csv}")
    return df

if __name__ == "__main__":
    generate_dataset()

