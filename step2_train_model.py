"""
STEP 2 - Baseline glucose forecaster (runs on CPU, no GPU needed).

What it does:
  1. Reads the sensor files made in step 1.
  2. Cuts them into "windows": look at the past 6 hours -> predict the next 2 hours of glucose.
  3. Trains a small LSTM (a model that reads a sequence in order).
  4. Compares it with a "lazy guess" (glucose stays flat) and prints the errors.
  5. Tests it on the 10 hard patients (T01-T10) that it never saw in training.
  6. Saves the model to models/ and a picture to plots/.

Run:  python step2_train_model.py
Needs: numpy, pandas, torch, matplotlib
"""
import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")           # draw pictures into files (no pop-up window needed)
import matplotlib.pyplot as plt

# ----------------------------- SETTINGS ------------------------------------
DATA = "data"
HISTORY = 72        # 72 readings x 5 min = 6 hours looked at
HORIZON = 24        # 24 readings x 5 min = 2 hours predicted
STRIDE = 3          # take a window every 3 readings (keeps training fast)
EPOCHS = 12
BATCH = 256
LR = 2e-3
SEED = 42
FEATURES = ["glucose_mgdl", "steps", "heart_rate", "hrv_ms", "sleeping", "carbs_g", "meal_gi"]
# how to scale each column to a small number (so the model learns easily)
SCALE = {"glucose_mgdl": 200.0, "steps": 500.0, "heart_rate": 100.0, "hrv_ms": 60.0,
         "sleeping": 1.0, "carbs_g": 100.0, "meal_gi": 100.0}

torch.manual_seed(SEED)
np.random.seed(SEED)
torch.set_num_threads(max(1, os.cpu_count() or 1))


# ----------------------------- DATA ----------------------------------------
def load_patient(pid):
    df = pd.read_csv(f"{DATA}/sensors/{pid}.csv", parse_dates=["timestamp"])
    x = np.stack([df[c].values / SCALE[c] for c in FEATURES], axis=1).astype(np.float32)
    # time of day as a circle (so 23:55 is close to 00:05)
    mins = (df["timestamp"].dt.hour * 60 + df["timestamp"].dt.minute).values
    tod = np.stack([np.sin(2 * np.pi * mins / 1440), np.cos(2 * np.pi * mins / 1440)], axis=1)
    x = np.concatenate([x, tod.astype(np.float32)], axis=1)
    return x, df["glucose_mgdl"].values.astype(np.float32)


def make_windows(x, g, start, end):
    """Cut rows [start, end) into (past 6h inputs, next 2h glucose change) pairs."""
    X, Y, last = [], [], []
    for i in range(start + HISTORY, end - HORIZON, STRIDE):
        X.append(x[i - HISTORY:i])
        future = g[i:i + HORIZON]
        now = g[i - 1]
        Y.append((future - now) / 100.0)      # predict CHANGE from now (easier to learn)
        last.append(now)
    return np.array(X), np.array(Y, dtype=np.float32), np.array(last, dtype=np.float32)


def build_sets(patient_ids, split_rows=None):
    Xs, Ys, Ls = [], [], []
    for pid in patient_ids:
        x, g = load_patient(pid)
        a, b = (0, len(g)) if split_rows is None else split_rows
        X, Y, L = make_windows(x, g, a, b)
        Xs.append(X); Ys.append(Y); Ls.append(L)
    return np.concatenate(Xs), np.concatenate(Ys), np.concatenate(Ls)


# ----------------------------- MODEL ---------------------------------------
class Forecaster(nn.Module):
    def __init__(self, n_in, hidden=64):
        super().__init__()
        self.lstm = nn.LSTM(n_in, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden, 64), nn.ReLU(), nn.Linear(64, HORIZON))

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1])          # use the last time step's summary


def rmse_by_horizon(pred_change, true_change):
    """RMSE in mg/dL at 30, 60 and 120 minutes ahead."""
    err = (pred_change - true_change) * 100.0
    return {f"{m}min": float(np.sqrt(np.mean(err[:, m // 5 - 1] ** 2))) for m in (30, 60, 120)}


def predict(model, X):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), 2048):
            out.append(model(torch.from_numpy(X[i:i + 2048])).numpy())
    return np.concatenate(out)


# ----------------------------- MAIN ----------------------------------------
def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    patients = pd.read_csv(f"{DATA}/patients.csv")
    train_ids = patients[patients.group == "train"].patient_id.tolist()
    test_ids = patients[patients.group == "test"].patient_id.tolist()
    n_rows = len(pd.read_csv(f"{DATA}/sensors/{train_ids[0]}.csv"))
    cut = int(n_rows * 0.8)       # first 80% of days = train, last 20% = validation

    Xtr, Ytr, _ = build_sets(train_ids, (0, cut))
    Xva, Yva, Lva = build_sets(train_ids, (cut, n_rows))
    print(f"Training windows: {len(Xtr)} | validation windows: {len(Xva)} | inputs per step: {Xtr.shape[2]}")

    model = Forecaster(Xtr.shape[2])
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()
    Xt, Yt = torch.from_numpy(Xtr), torch.from_numpy(Ytr)

    best = 1e9
    t0 = time.time()
    for ep in range(1, EPOCHS + 1):
        model.train()
        perm = torch.randperm(len(Xt))
        total = 0.0
        for i in range(0, len(Xt), BATCH):
            idx = perm[i:i + BATCH]
            opt.zero_grad()
            loss = loss_fn(model(Xt[idx]), Yt[idx])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * len(idx)
        va = rmse_by_horizon(predict(model, Xva), Yva)
        print(f"epoch {ep:2d}/{EPOCHS}  train loss {total / len(Xt):.4f}  "
              f"validation RMSE 30/60/120 min: {va['30min']:.1f} / {va['60min']:.1f} / {va['120min']:.1f} mg/dL  "
              f"({time.time() - t0:.0f}s)")
        if va["120min"] < best:
            best = va["120min"]
            torch.save(model.state_dict(), "models/baseline_lstm.pt")
    model.load_state_dict(torch.load("models/baseline_lstm.pt"))

    # ---- compare with the lazy guess: "glucose stays exactly where it is now" ----
    lazy = rmse_by_horizon(np.zeros_like(Yva), Yva)
    mine = rmse_by_horizon(predict(model, Xva), Yva)
    print("\nVALIDATION (patients seen in training, but later days)")
    print(f"  lazy guess RMSE 30/60/120 min: {lazy['30min']:.1f} / {lazy['60min']:.1f} / {lazy['120min']:.1f}")
    print(f"  our model  RMSE 30/60/120 min: {mine['30min']:.1f} / {mine['60min']:.1f} / {mine['120min']:.1f}")

    # ---- the 10 hard test patients (never seen) ----
    print("\nHARD TEST PATIENTS (never seen)   RMSE in mg/dL, smaller is better")
    print("  patient   lazy@60  model@60   lazy@120  model@120")
    rows = []
    for pid in test_ids:
        X, Y, _ = build_sets([pid])
        l, m = rmse_by_horizon(np.zeros_like(Y), Y), rmse_by_horizon(predict(model, X), Y)
        print(f"  {pid}      {l['60min']:6.1f}   {m['60min']:6.1f}    {l['120min']:7.1f}   {m['120min']:7.1f}")
        rows.append(dict(patient_id=pid, lazy_60=l["60min"], model_60=m["60min"],
                         lazy_120=l["120min"], model_120=m["120min"]))
    pd.DataFrame(rows).to_csv("models/test_results_step2.csv", index=False)

    # ---- one picture: forecasts vs reality for a hard patient ----
    pid = "T03"
    x, g = load_patient(pid)
    fig, ax = plt.subplots(figsize=(11, 4))
    start = 3 * 288 + HISTORY
    ax.plot(np.arange(start - HISTORY, start + 288) * 5 / 60, g[start - HISTORY:start + 288], color="gray", label="real glucose")
    for k in range(0, 288 - HORIZON, 36):         # a new 2-hour forecast every 3 hours
        i = start + k
        pred = g[i - 1] + predict(model, x[None, i - HISTORY:i])[0] * 100
        ax.plot(np.arange(i, i + HORIZON) * 5 / 60, pred, color="tab:red")
    ax.axhline(180, color="orange", ls="--", lw=0.8)
    ax.axhline(70, color="blue", ls="--", lw=0.8)
    ax.set_xlabel("hours since start"); ax.set_ylabel("glucose (mg/dL)")
    ax.set_title(f"{pid}: real glucose (gray) and 2-hour forecasts (red)")
    ax.legend(); fig.tight_layout(); fig.savefig("plots/forecast_T03.png", dpi=130)
    print("\nSaved: models/baseline_lstm.pt, models/test_results_step2.csv, plots/forecast_T03.png")


if __name__ == "__main__":
    main()
