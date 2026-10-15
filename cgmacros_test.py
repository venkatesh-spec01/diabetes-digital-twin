"""
Real-data check on CGMacros (PhysioNet): 45 adults, 10 days each, with a
Dexcom G6 sensor, Fitbit, logged meals and blood tests. The models trained on
synthetic data are run unchanged on these people.
Dataset: https://physionet.org/content/cgmacros/1.0.0/  (CC BY-NC-SA 4.0, not stored in this repo)

Setup: download and unzip it into cgmacros_data/ (the script looks for bio.csv).
Run: python cgmacros_test.py

CGMacros has no steps, HRV, sleep or meal GI, so these are estimated:
  steps  from Fitbit METs (about 40 steps/min per MET above rest)
  HRV    fixed at 40 ms
  sleep  low activity between 23:00 and 07:00
  GI     60 for every meal
  medicines, years with diabetes, family history: set to the training average
Clock times in CGMacros are shifted for privacy, so time of day is approximate.
Saves models/real_data_step10.csv and plots/real_data_example.png
"""
import glob
import os
import numpy as np
import pandas as pd

from train_baseline import DATA, HISTORY, HORIZON, STRIDE, SCALE, rmse_by_horizon
from fusion_model import EHR_COLUMNS, load_ehr

ROOT = "cgmacros_data"
HIGH, LOW = 180.0, 70.0
MODEL_FILE = "models/fusion_cf_blend75.pt"   # the model the dashboard uses


# ----------------------------- find the files -----------------------------
def find_root():
    hits = glob.glob(os.path.join(ROOT, "**", "bio.csv"), recursive=True)
    if not hits:
        raise SystemExit(f"Could not find bio.csv inside '{ROOT}/'. See the Setup notes at the top of this file.")
    return os.path.dirname(hits[0])


def col(df, *keys):
    """Find a column by keywords (case and spaces ignored): exact name, then starts-with, then contains."""
    norm = {c: str(c).lower().replace(" ", "") for c in df.columns}
    for test in (lambda n: n == keys[0], lambda n: n.startswith(keys[0]) and all(k in n for k in keys),
                 lambda n: all(k in n for k in keys)):
        for c, n in norm.items():
            if test(n):
                return c
    return None


# ----------------------------- health records -----------------------------
def read_bio(root, subject_ids):
    bio = pd.read_csv(os.path.join(root, "bio.csv"))
    sid = col(bio, "subject")
    if sid is not None:
        bio["sid"] = bio[sid].astype(str).str.extract(r"(\d+)")[0].astype(int)
    else:
        # the official notebook assumes rows are in the same order as the participant folders
        print("Note: bio.csv has no subject column, matching rows to folders in sorted order.")
        bio = bio.iloc[:len(subject_ids)].copy()
        bio["sid"] = sorted(subject_ids)[:len(bio)]
    bio = bio.set_index("sid")

    def num(c):
        return pd.to_numeric(bio[c].astype(str).str.extract(r"([-\d.]+)")[0], errors="coerce") if c else np.nan

    weight_lb, height_in = num(col(bio, "weight")), num(col(bio, "height"))
    out = pd.DataFrame(index=bio.index)
    out["age"] = num(col(bio, "age"))
    g = bio[col(bio, "gender")].astype(str).str.upper().str[0] if col(bio, "gender") else "M"
    out["sex"] = np.where(g == "F", "F", "M")
    out["bmi"] = (weight_lb * 0.453592) / (height_in * 0.0254) ** 2
    out["hba1c"] = num(col(bio, "a1c"))
    out["fasting_glucose"] = num(col(bio, "fasting", "glu"))
    out["ldl"] = num(col(bio, "ldl"))
    out["triglycerides"] = num(col(bio, "triglycerides"))
    out["group_real"] = np.select([out.hba1c < 5.7, out.hba1c <= 6.4], ["healthy", "pre-diabetes"], "T2D")
    return out


def ehr_for(real):
    """Scale real records exactly like the training records; unknown fields = training average."""
    patients = pd.read_csv(f"{DATA}/patients.csv")
    train_ids = patients[patients.group == "train"].patient_id.tolist()
    train = patients[patients.group == "train"]
    rows = []
    for sid, r in real.iterrows():
        row = {c: (train[c].mean() if c != "sex" else "M") for c in EHR_COLUMNS}
        for c in ["age", "sex", "bmi", "hba1c", "fasting_glucose", "ldl", "triglycerides"]:
            if pd.notna(r[c]):
                row[c] = r[c]
        row["patient_id"] = f"CGM{sid:03d}"
        rows.append(row)
    both = pd.concat([patients, pd.DataFrame(rows)], ignore_index=True)
    return load_ehr(both, train_ids)


# ----------------------------- sensor stream ------------------------------
def fill_short_gaps(s, max_len):
    """Fill gaps of up to max_len missing readings; longer gaps stay missing (those windows are skipped)."""
    filled = s.interpolate(limit_area="inside")
    run = s.isna().groupby(s.notna().cumsum()).transform("sum")
    return filled.where(s.notna() | (run <= max_len))


def read_stream(path):
    """One participant: 1-minute rows -> 5-minute rows with our 10 input signals."""
    df = pd.read_csv(path)
    df["t"] = pd.to_datetime(df[col(df, "timestamp")], errors="coerce")
    df = df.dropna(subset=["t"]).set_index("t").sort_index()
    dex, lib = col(df, "dexcom"), col(df, "libre")
    hr = "HR" if "HR" in df.columns else col(df, "heart")
    mets = col(df, "mets")
    carbs, amount, meal = col(df, "carbs"), col(df, "amount"), col(df, "meal", "type")

    def five(c):
        return pd.to_numeric(df[c], errors="coerce").resample("5min").mean() if c else None
    g5 = five(dex)
    if (g5 is None or g5.notna().mean() < 0.5) and lib:      # Dexcom mostly missing -> use Libre (every 15 min)
        g5 = five(lib)
    m = pd.to_numeric(df[mets], errors="coerce") / 10.0 if mets else pd.Series(1.0, index=df.index)
    steps_min = ((m - 1.0) * 40).clip(0, 180)
    c = pd.to_numeric(df[carbs], errors="coerce").fillna(0) if carbs else 0
    if amount:
        c = c * (pd.to_numeric(df[amount], errors="coerce").fillna(100) / 100.0)

    r = pd.DataFrame({
        "glucose": fill_short_gaps(g5, 3),
        "steps": steps_min.resample("5min").sum(min_count=1),
        "hr": pd.to_numeric(df[hr], errors="coerce").resample("5min").mean() if hr else np.nan,
        "mets": m.resample("5min").mean(),
        "carbs": c.resample("5min").sum() if carbs else 0.0,
    })
    r["meal"] = (df[meal].notna().resample("5min").max() if meal else False)
    r["steps"] = r.steps.fillna(0)
    r["hr"] = r.hr.interpolate(limit_area="inside").ffill().bfill().fillna(75)
    hour = r.index.hour + r.index.minute / 60
    r["sleeping"] = (((hour >= 23) | (hour < 7)) & (r.mets.fillna(1) <= 1.1)).astype(float)
    r["gi"] = np.where(r.carbs > 0, 60.0, 0.0)

    # walk flag, like training: active for 10+ minutes in the 15-40 minutes after a meal
    walk = np.zeros(len(r), dtype=np.float32)
    active = (r.mets.fillna(1) >= 2.5).values
    for i in np.where(r.carbs.values > 0)[0]:
        if active[i + 3:i + 8].sum() >= 2:
            walk[i + 3:i + 5] = 1.0

    mins = hour.values * 60
    x = np.stack([
        r.glucose.values / SCALE["glucose_mgdl"], r.steps.values / SCALE["steps"],
        r.hr.values / SCALE["heart_rate"], np.full(len(r), 40.0) / SCALE["hrv_ms"],
        r.sleeping.values, r.carbs.values / SCALE["carbs_g"], r.gi.values / SCALE["meal_gi"],
        np.sin(2 * np.pi * mins / 1440), np.cos(2 * np.pi * mins / 1440), walk,
    ], axis=1).astype(np.float32)
    return x, r.glucose.values.astype(np.float32), r


def windows(x, g):
    """Same windows as training (6 h in, 2 h out, every 15 min), skipping gaps in the CGM."""
    X, Y, L, idx = [], [], [], []
    for i in range(HISTORY, len(g) - HORIZON, STRIDE):
        past, future = g[i - HISTORY:i], g[i:i + HORIZON]
        if np.isnan(past).any() or np.isnan(future).any():
            continue
        X.append(x[i - HISTORY:i])
        Y.append((future - g[i - 1]) / 100.0)
        L.append(g[i - 1])
        idx.append(i)
    if not X:
        return None
    return np.array(X), np.array(Y, dtype=np.float32), np.array(L, dtype=np.float32), np.array(idx)


def alert_scores(forecast, future, now, margin=0):
    out = {}
    for kind in ("HIGH", "LOW"):
        if kind == "HIGH":
            mask, event, alert = now <= HIGH, future.max(1) > HIGH, forecast.max(1) > HIGH - margin
        else:
            mask, event, alert = now >= LOW, future.min(1) < LOW, forecast.min(1) < LOW + margin
        a, e = alert[mask], event[mask]
        tp, fp, fn = int((a & e).sum()), int((a & ~e).sum()), int((~a & e).sum())
        p = tp / (tp + fp) if tp + fp else np.nan
        r = tp / (tp + fn) if tp + fn else np.nan
        f1 = 2 * p * r / (p + r) if (p + r) and not np.isnan(p + r) else np.nan
        out.update({f"{kind}_events": int(e.sum()), f"{kind}_precision": p, f"{kind}_recall": r, f"{kind}_F1": f1})
    return out


# ----------------------------- main ----------------------------------------
def main():
    import torch
    from train_baseline import Forecaster, predict as predict_seq
    from fusion_model import FusionForecaster, predict_fused

    root = find_root()
    files = {}
    for f in glob.glob(os.path.join(root, "*", "*.csv")):
        num = "".join(ch for ch in os.path.basename(f) if ch.isdigit())
        if os.path.basename(f).lower().startswith("cgmacros") and num:
            files[int(num)] = f
    print(f"Found {len(files)} participant files in {root}")
    real = read_bio(root, list(files))
    ehr = ehr_for(real)

    base = Forecaster(10)
    base.load_state_dict(torch.load("models/baseline_lstm.pt"))
    fuse = FusionForecaster(10, ehr.shape[1])
    fuse.load_state_dict(torch.load(MODEL_FILE))

    per_group = {}
    example = None
    for sid, path in sorted(files.items()):
        if sid not in real.index:
            continue
        x, g, r = read_stream(path)
        w = windows(x, g)
        if w is None:
            print(f"  {sid:03d}: no usable windows (CGM gaps), skipped")
            continue
        X, Y, L, idx = w
        S = np.repeat(ehr.loc[f"CGM{sid:03d}"].values.astype(np.float32)[None], len(X), axis=0)
        preds = {
            "no change (flat)": np.zeros_like(Y),
            "trend": ((X[:, -1, 0] - X[:, -7, 0]) * 200 / 6)[:, None] * np.arange(1, HORIZON + 1) / 100,
            "sensor-only LSTM": predict_seq(base, X),
            "fusion twin (dashboard model)": predict_fused(fuse, X, S),
        }
        grp = real.loc[sid, "group_real"]
        d = per_group.setdefault(grp, {"Y": [], "L": [], "P": {k: [] for k in preds}, "people": 0})
        d["Y"].append(Y); d["L"].append(L); d["people"] += 1
        for k, v in preds.items():
            d["P"][k].append(v)
        if grp == "T2D" and (example is None or len(X) > example[1]):
            example = (sid, len(X), g, idx, preds["fusion twin (dashboard model)"], L)

    rows = []
    for grp in ["T2D", "pre-diabetes", "healthy"]:
        if grp not in per_group:
            continue
        d = per_group[grp]
        Y, L = np.concatenate(d["Y"]), np.concatenate(d["L"])
        future = L[:, None] + Y * 100
        print(f"\n{grp.upper()}: {d['people']} people, {len(Y)} windows")
        print("  model                                    RMSE 30 / 60 / 120 min   HIGH F1 (P/R)        LOW F1 (P/R)")
        for k in d["P"]:
            P = np.concatenate(d["P"][k])
            e = rmse_by_horizon(P, Y)
            fc = L[:, None] + P * 100
            for margin in ([0, 15] if k.startswith("fusion") else [0]):
                a = alert_scores(fc, future, L, margin)
                name = k + (" sensitive" if margin else "")
                print(f"  {name:40s} {e['30min']:5.1f} / {e['60min']:5.1f} / {e['120min']:5.1f}    "
                      f"{a['HIGH_F1']:.2f} ({a['HIGH_precision']:.2f}/{a['HIGH_recall']:.2f})    "
                      f"{a['LOW_F1']:.2f} ({a['LOW_precision']:.2f}/{a['LOW_recall']:.2f})")
                rows.append(dict(group=grp, people=d["people"], windows=len(Y), model=name,
                                 rmse_30=e["30min"], rmse_60=e["60min"], rmse_120=e["120min"], **a))
    os.makedirs("plots", exist_ok=True)
    pd.DataFrame(rows).round(3).to_csv("models/real_data_step10.csv", index=False)
    print("\nFor comparison, the fusion twin on SYNTHETIC test patients: 2-hour RMSE about 18 mg/dL.")

    if example:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        sid, _, g, idx, P, L = example
        start = idx[len(idx) // 3]
        span = range(start - HISTORY, min(start + 288, len(g)))
        fig, ax = plt.subplots(figsize=(11, 4))
        ax.plot(np.array(span) * 5 / 60, g[list(span)], color="gray", label="real glucose (CGMacros)")
        for j, i in enumerate(idx):
            if start <= i < start + 288 - HORIZON and (i - start) % 36 == 0:
                ax.plot(np.arange(i, i + HORIZON) * 5 / 60, L[j] + P[j] * 100, color="tab:red")
        ax.axhline(180, color="orange", ls="--", lw=0.8)
        ax.axhline(70, color="blue", ls="--", lw=0.8)
        ax.set_xlabel("hours"); ax.set_ylabel("glucose (mg/dL)")
        ax.set_title(f"Real T2D participant {sid:03d}: real glucose (gray) and 2-hour twin forecasts (red)")
        ax.legend(); fig.tight_layout(); fig.savefig("plots/real_data_example.png", dpi=130)
    print("Saved: models/real_data_step10.csv, plots/real_data_example.png")


if __name__ == "__main__":
    main()
