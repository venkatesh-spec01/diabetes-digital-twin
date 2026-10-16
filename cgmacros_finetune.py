# fine-tunes the twin on real people (CGMacros) and tests it on other real people.
# people are split by person, never by time window (seed 42):
#   train about 29 people, valid about 6 (only to pick the best round),
#   test about 10 people (looked at only once at the end).
# each group (T2D, pre-diabetes, healthy) is split the same way.
# to avoid forgetting the synthetic training, we mix synthetic windows into every round,
# so the what-if part keeps working. the dashboard model is not overwritten.
# needs the same cgmacros_data folder as cgmacros_test.py
# saves the new model, the split and the before/after table in the models folder.
# run: python cgmacros_finetune.py (takes a few minutes on cpu)
import glob
import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from train_baseline import DATA, HORIZON, BATCH, SEED, rmse_by_horizon
from fusion_model import load_ehr, build_fused, FusionForecaster, predict_fused
from cgmacros_test import (find_root, read_bio, ehr_for, read_stream, windows,
                                  alert_scores)

# model we start from (dashboard model)
BASE_MODEL = "models/fusion_cf_blend75.pt"
# where the fine-tuned model is saved
NEW_MODEL = "models/fusion_real_ft1b.pt"
EPOCHS = 8
# small learning rate so it forgets less
LR = 1e-4
# how many synthetic windows to mix in
SYN_RATIO = 3.0
# people held out for test and validation
N_TEST, N_VALID = {"T2D": 3, "pre-diabetes": 4, "healthy": 3}, 2


# splits people into train, valid and test
def make_split(real, ids):
    # fixed seed so the split repeats
    rng = np.random.RandomState(SEED)
    split = {}
    for grp in ["T2D", "pre-diabetes", "healthy"]:
        people = sorted(i for i in ids if real.loc[i, "group_real"] == grp)
        rng.shuffle(people)
        n_te, n_va = N_TEST[grp], N_VALID
        for k, sid in enumerate(people):
            split[sid] = "test" if k < n_te else "valid" if k < n_te + n_va else "train"
    return split


# loads windows for every real person
def load_people(files, real, ehr, split):
    out = {}
    for sid, path in sorted(files.items()):
        if sid not in split:
            continue
        x, g, _ = read_stream(path)
        w = windows(x, g)
        if w is None:
            print(f"  {sid:03d}: no usable windows, skipped")
            continue
        X, Y, L, _ = w
        S = np.repeat(ehr.loc[f"CGM{sid:03d}"].values.astype(np.float32)[None], len(X), axis=0)
        out[sid] = (X, Y, L, S)
    return out


# joins windows of people in one group
def stack(people, split, which):
    parts = [v for sid, v in people.items() if split[sid] == which]
    return (np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts]),
            np.concatenate([p[2] for p in parts]), np.concatenate([p[3] for p in parts]))


# 2 hour error for quick checks
def rmse120(model, X, S, Y):
    return rmse_by_horizon(predict_fused(model, X, S), Y)["120min"]


# prints before and after on test people
def report(models, people, split, real):
    rows = []
    for grp in ["T2D", "pre-diabetes", "healthy", "ALL"]:
        ids = [s for s in people if split[s] == "test" and (grp == "ALL" or real.loc[s, "group_real"] == grp)]
        if not ids:
            continue
        X = np.concatenate([people[s][0] for s in ids])
        Y = np.concatenate([people[s][1] for s in ids])
        L = np.concatenate([people[s][2] for s in ids])
        S = np.concatenate([people[s][3] for s in ids])
        future = L[:, None] + Y * 100
        print(f"\nTEST people - {grp}: {len(ids)} people, {len(Y)} windows")
        print("  model                         RMSE 30 / 60 / 120 min   HIGH recall (default / sensitive)   HIGH F1")
        preds = {"no change (flat)": np.zeros_like(Y)}
        for name, m in models.items():
            preds[name] = predict_fused(m, X, S)
        for name, P in preds.items():
            e = rmse_by_horizon(P, Y)
            fc = L[:, None] + P * 100
            a0, a15 = alert_scores(fc, future, L, 0), alert_scores(fc, future, L, 15)
            print(f"  {name:28s} {e['30min']:5.1f} / {e['60min']:5.1f} / {e['120min']:5.1f}"
                  f"        {a0['HIGH_recall']:.2f} / {a15['HIGH_recall']:.2f}"
                  f"                        {a0['HIGH_F1']:.2f}")
            rows.append(dict(group=grp, people=len(ids), windows=len(Y), model=name,
                             rmse_30=e["30min"], rmse_60=e["60min"], rmse_120=e["120min"],
                             high_recall_default=a0["HIGH_recall"], high_recall_sensitive=a15["HIGH_recall"],
                             high_F1_default=a0["HIGH_F1"], high_F1_sensitive=a15["HIGH_F1"]))
    return rows


# checks the what-if part still works after training
def remedy_check(models):
    from whatif_finetune import truth_table, test_forecast_error
    patients = pd.read_csv(f"{DATA}/patients.csv")
    train_ids = patients[patients.group == "train"].patient_id.tolist()
    test_ids = patients[patients.group == "test"].patient_id.tolist()
    ehr = load_ehr(patients, train_ids)
    pat = patients.set_index("patient_id")
    meals = pd.read_csv(f"{DATA}/meals.csv", parse_dates=["timestamp"])
    print("\n=== Safety check on the 10 SYNTHETIC test patients (should stay about the same) ===")
    for name, m in models.items():
        e = test_forecast_error(m, test_ids, ehr)
        print(f"\n{name}: synthetic 2-hour RMSE {e.mean():.2f} mg/dL.  What-if remedies vs the true answer:")
        print(truth_table(m, test_ids, ehr, pat, meals).to_string())


# splits people, fine-tunes, tests, saves results
def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    root = find_root()
    files = {}
    for f in glob.glob(os.path.join(root, "*", "*.csv")):
        num = "".join(ch for ch in os.path.basename(f) if ch.isdigit())
        if os.path.basename(f).lower().startswith("cgmacros") and num:
            files[int(num)] = f
    real = read_bio(root, list(files))
    ehr_real = ehr_for(real)
    split = make_split(real, [s for s in files if s in real.index])
    pd.DataFrame({"person": list(split), "group": [real.loc[s, "group_real"] for s in split],
                  "role": list(split.values())}).to_csv("models/step11_split.csv", index=False)
    print("Split:", pd.Series(split).value_counts().to_dict(), "(saved to models/step11_split.csv)")

    people = load_people(files, real, ehr_real, split)
    Xtr, Ytr, _, Str = stack(people, split, "train")
    Xva, Yva, _, Sva = stack(people, split, "valid")
    print(f"Real windows: train {len(Xtr)}, valid {len(Xva)}")

    patients = pd.read_csv(f"{DATA}/patients.csv")
    syn_ids = patients[patients.group == "train"].patient_id.tolist()
    ehr_syn = load_ehr(patients, syn_ids)
    n_rows = len(pd.read_csv(f"{DATA}/sensors/{syn_ids[0]}.csv"))
    Xs, Ss, Ys = build_fused(syn_ids, ehr_syn, (0, int(n_rows * 0.8)))
    # pick random synthetic windows to mix in
    pick = np.random.choice(len(Xs), min(len(Xs), int(SYN_RATIO * len(Xtr))), replace=False)
    X = torch.from_numpy(np.concatenate([Xtr, Xs[pick]]))
    S = torch.from_numpy(np.concatenate([Str, Ss[pick]]))
    Y = torch.from_numpy(np.concatenate([Ytr, Ys[pick]]))
    del Xs, Ss, Ys
    print(f"Training windows: {len(Xtr)} real + {len(pick)} synthetic")

    # keep the original model for comparison
    before = FusionForecaster(10, S.shape[1])
    before.load_state_dict(torch.load(BASE_MODEL))
    model = FusionForecaster(10, S.shape[1])
    model.load_state_dict(torch.load(BASE_MODEL))
    # validation error before any training
    best = rmse120(model, Xva, Sva, Yva)
    best_state = {k: v.clone() for k, v in model.state_dict().items()}
    print(f"\nVALID people, 2-hour RMSE before any training: {best:.2f} mg/dL")

    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    loss_fn = nn.MSELoss()
    t0 = time.time()
    for ep in range(1, EPOCHS + 1):
        model.train()
        perm = torch.randperm(len(X))
        for i in range(0, len(X), BATCH):
            idx = perm[i:i + BATCH]
            opt.zero_grad()
            loss = loss_fn(model(X[idx], S[idx] + 0.05 * torch.randn_like(S[idx])), Y[idx])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        v = rmse120(model, Xva, Sva, Yva)
        mark = ""
        if v < best:
            best, mark = v, "  <- best so far"
            best_state = {k: t.clone() for k, t in model.state_dict().items()}
        print(f"round {ep}/{EPOCHS}: VALID 2-hour RMSE {v:.2f}{mark}   ({time.time() - t0:.0f}s)")

    # go back to the best round
    model.load_state_dict(best_state)
    torch.save(best_state, NEW_MODEL)
    print(f"\nBest round on VALID people: {best:.2f} mg/dL. Saved {NEW_MODEL}")

    rows = report({"before (dashboard model)": before, "after fine-tuning": model}, people, split, real)
    pd.DataFrame(rows).round(3).to_csv("models/real_finetune_step11b.csv", index=False)
    print("\nSaved: models/real_finetune_step11.csv")
    try:
        remedy_check({"before": before, "after": model})
    except Exception as e:
        print(f"\n(Synthetic safety check could not run: {e})")
    print("\nNote: the TEST group is only about 10 people, so treat small differences as noise.")


if __name__ == "__main__":
    main()
