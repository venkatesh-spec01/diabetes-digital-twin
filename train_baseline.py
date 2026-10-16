# baseline model: a small LSTM that reads only the sensor data.
# it looks at the last 6 hours and forecasts the next 2 hours of glucose.
# it is compared with a lazy guess (glucose stays the same) and tested
# on the 10 hard patients it never saw in training.
# the saved model and a plot go to the models and plots folders.
# run: python train_baseline.py
import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = "data"
# 72 readings = last 6 hours of data
HISTORY = 72
# 24 readings = next 2 hours to forecast
HORIZON = 24
# move 3 steps between windows
STRIDE = 3
EPOCHS = 12
BATCH = 256
LR = 2e-3
SEED = 42
# sensor columns the model reads as input
FEATURES = ["glucose_mgdl", "steps", "heart_rate", "hrv_ms", "sleeping", "carbs_g", "meal_gi"]
# divide each column to keep values small
SCALE = {"glucose_mgdl": 200.0, "steps": 500.0, "heart_rate": 100.0, "hrv_ms": 60.0,
         "sleeping": 1.0, "carbs_g": 100.0, "meal_gi": 100.0}

# same seed so results repeat
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.set_num_threads(max(1, os.cpu_count() or 1))


# reads one patient's csv and makes model input
def load_patient(pid):
    df = pd.read_csv(f"{DATA}/sensors/{pid}.csv", parse_dates=["timestamp"])
    x = np.stack([df[c].values / SCALE[c] for c in FEATURES], axis=1).astype(np.float32)
    mins = (df["timestamp"].dt.hour * 60 + df["timestamp"].dt.minute).values
    # time of day as sine and cosine
    tod = np.stack([np.sin(2 * np.pi * mins / 1440), np.cos(2 * np.pi * mins / 1440)], axis=1)
    # flag for walking after meals
    walk = np.zeros((len(df), 1), dtype=np.float32)
    meals = pd.read_csv(f"{DATA}/meals.csv", parse_dates=["timestamp"])
    meals = meals[(meals.patient_id == pid) & (meals.walked_after == 1)]
    for ts in meals["timestamp"]:
        t0 = int(round((ts - df["timestamp"].iloc[0]).total_seconds() / 300))
        walk[t0 + 3:t0 + 5, 0] = 1.0
    x = np.concatenate([x, tod.astype(np.float32), walk], axis=1)
    return x, df["glucose_mgdl"].values.astype(np.float32)
    

# cuts data into past 6h and next 2h pieces
def make_windows(x, g, start, end):
    X, Y, last = [], [], []
    for i in range(start + HISTORY, end - HORIZON, STRIDE):
        X.append(x[i - HISTORY:i])
        future = g[i:i + HORIZON]
        now = g[i - 1]
        # target is change from current glucose
        Y.append((future - now) / 100.0)
        last.append(now)
    return np.array(X), np.array(Y, dtype=np.float32), np.array(last, dtype=np.float32)


# makes windows for many patients together
def build_sets(patient_ids, split_rows=None):
    Xs, Ys, Ls = [], [], []
    for pid in patient_ids:
        x, g = load_patient(pid)
        a, b = (0, len(g)) if split_rows is None else split_rows
        X, Y, L = make_windows(x, g, a, b)
        Xs.append(X); Ys.append(Y); Ls.append(L)
    return np.concatenate(Xs), np.concatenate(Ys), np.concatenate(Ls)


# small LSTM that predicts the next 2 hours
class Forecaster(nn.Module):
    def __init__(self, n_in, hidden=64):
        super().__init__()
        self.lstm = nn.LSTM(n_in, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden, 64), nn.ReLU(), nn.Linear(64, HORIZON))

    # LSTM reads the history, last output goes to head
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1])


# error in mg/dL at 30, 60 and 120 minutes
def rmse_by_horizon(pred_change, true_change):
    err = (pred_change - true_change) * 100.0
    return {f"{m}min": float(np.sqrt(np.mean(err[:, m // 5 - 1] ** 2))) for m in (30, 60, 120)}


# runs the model on the data in batches
def predict(model, X):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), 2048):
            out.append(model(torch.from_numpy(X[i:i + 2048])).numpy())
    return np.concatenate(out)


# trains, compares with lazy guess, tests, saves
def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    patients = pd.read_csv(f"{DATA}/patients.csv")
    train_ids = patients[patients.group == "train"].patient_id.tolist()
    test_ids = patients[patients.group == "test"].patient_id.tolist()
    n_rows = len(pd.read_csv(f"{DATA}/sensors/{train_ids[0]}.csv"))
    # first 80 percent train, last 20 validation
    cut = int(n_rows * 0.8)

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
        # keep the model with best 2 hour error
        if va["120min"] < best:
            best = va["120min"]
            torch.save(model.state_dict(), "models/baseline_lstm.pt")
    model.load_state_dict(torch.load("models/baseline_lstm.pt"))

    # lazy guess means glucose stays the same
    lazy = rmse_by_horizon(np.zeros_like(Yva), Yva)
    mine = rmse_by_horizon(predict(model, Xva), Yva)
    print("\nVALIDATION (patients seen in training, but later days)")
    print(f"  lazy guess RMSE 30/60/120 min: {lazy['30min']:.1f} / {lazy['60min']:.1f} / {lazy['120min']:.1f}")
    print(f"  our model  RMSE 30/60/120 min: {mine['30min']:.1f} / {mine['60min']:.1f} / {mine['120min']:.1f}")

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

    # plot one test patient as an example
    pid = "T03"
    x, g = load_patient(pid)
    fig, ax = plt.subplots(figsize=(11, 4))
    start = 3 * 288 + HISTORY
    ax.plot(np.arange(start - HISTORY, start + 288) * 5 / 60, g[start - HISTORY:start + 288], color="gray", label="real glucose")
    for k in range(0, 288 - HORIZON, 36):
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
