"""
STEP 5 - What-if remedy engine.

For each real meal of the 10 unseen test patients, look at the moment 30 minutes after
the meal, change ONE thing (walk, smaller portion, lower-GI dish), and ask the fusion
model how the next 2 hours of glucose would change.

Run:  python step5_remedies.py
"""
import numpy as np
import pandas as pd
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from step2_train_model import DATA, HISTORY, HORIZON, load_patient
from step3_fusion import load_ehr, FusionForecaster, predict_fused

START = pd.Timestamp("2026-09-01 00:00")
AFTER = 6
HIGH = 180.0
REMEDIES = ["walk 10 min", "30% smaller portion", "low-GI swap (GI 50)", "walk + smaller portion"]


def make_variant(x, t0, gi, remedy):
	"""Return a modified copy of the 6-hour window ending 30 minutes after a meal."""
	i = t0 + AFTER
	w = x[i - HISTORY : i].copy()
	lo = i - HISTORY
	walk = remedy in ("walk 10 min", "walk + smaller portion")
	scale = 1.0
	if remedy in ("30% smaller portion", "walk + smaller portion"):
		scale *= 0.7
	if remedy == "low-GI swap (GI 50)":
		if gi <= 50:
			return None
		scale *= 50.0 / gi
		w[t0 - lo, 6] = 50.0 / 100.0
	if remedy in ("30% smaller portion", "walk + smaller portion"):
		w[t0 - lo, 5] *= 0.7
	if scale < 1.0:
		g0 = x[t0 - 1, 0]
		for a in range(t0, i):
			w[a - lo, 0] = g0 + (x[a, 0] - g0) * scale
	if walk:
		for a in (t0 + 3, t0 + 4):
			w[a - lo, 1] += 450.0 / 500.0
			w[a - lo, 2] += 27.0 / 100.0
			w[a - lo, 9] = 1.0
	return w


def forecast(model, W, s):
	S = np.repeat(s[None], len(W), axis=0).astype(np.float32)
	now = W[:, -1, 0] * 200.0
	return now[:, None] + predict_fused(model, W.astype(np.float32), S) * 100.0


def main():
	patients = pd.read_csv(f"{DATA}/patients.csv")
	train_ids = patients[patients.group == "train"].patient_id.tolist()
	test_ids = patients[patients.group == "test"].patient_id.tolist()
	ehr = load_ehr(patients, train_ids)
	meals = pd.read_csv(f"{DATA}/meals.csv", parse_dates=["timestamp"])

	model = FusionForecaster(10, ehr.shape[1])
	model.load_state_dict(torch.load("models/fusion_lstm.pt"))
	model.eval()

	rows = []
	for pid in test_ids:
		x, g = load_patient(pid)
		s = ehr.loc[pid].values.astype(np.float32)
		m = meals[(meals.patient_id == pid) & (meals.walked_after == 0)]
		base_W, info = [], []
		for _, r in m.iterrows():
			t0 = int(round((r.timestamp - START).total_seconds() / 300))
			i = t0 + AFTER
			if t0 < 1 or i < HISTORY or i + HORIZON > len(g):
				continue
			base_W.append(x[i - HISTORY : i].copy())
			info.append((t0, r.gi, r.meal, r.slot))
		if not base_W:
			continue
		f0 = forecast(model, np.array(base_W), s)
		peak0 = f0.max(axis=1)
		above0 = (f0 > HIGH).sum(axis=1) * 5.0
		for remedy in REMEDIES:
			W, keep = [], []
			for k, (t0, gi, meal, slot) in enumerate(info):
				w = make_variant(x, t0, gi, remedy)
				if w is not None:
					W.append(w)
					keep.append(k)
			if not W:
				continue
			f1 = forecast(model, np.array(W), s)
			peak1 = f1.max(axis=1)
			above1 = (f1 > HIGH).sum(axis=1) * 5.0
			for j, k in enumerate(keep):
				rows.append(dict(
					patient_id=pid, remedy=remedy, meal=info[k][2], slot=info[k][3],
					peak_before=peak0[k], peak_after=peak1[j], peak_drop=peak0[k] - peak1[j],
					min_above_before=above0[k], min_above_after=above1[j],
					min_above_drop=above0[k] - above1[j],
				))
	df = pd.DataFrame(rows)
	df.to_csv("models/remedy_results_step5.csv", index=False)

	print(f"Tested {df.groupby('remedy').size().max()} meals per remedy across {len(test_ids)} unseen patients\n")
	overall = df.groupby("remedy").agg(
		meals=("peak_drop", "size"),
		avg_peak_drop=("peak_drop", "mean"),
		avg_minutes_above_180_saved=("min_above_drop", "mean"),
		pct_meals_improved=("peak_drop", lambda v: 100 * (v > 0).mean()),
	)
	print("OVERALL (predicted by the model, higher = better)")
	print(overall.sort_values("avg_peak_drop", ascending=False).round(1).to_string())

	print("\nBEST REMEDY PER PATIENT (average predicted peak drop, mg/dL)")
	pv = df.groupby(["patient_id", "remedy"]).peak_drop.mean().unstack().round(1)
	pv["best"] = pv.idxmax(axis=1)
	print(pv.to_string())

	fig, ax = plt.subplots(figsize=(11, 4.5))
	w = 0.2
	pos = np.arange(len(pv))
	for k, rem in enumerate(REMEDIES):
		ax.bar(pos + (k - 1.5) * w, pv[rem].values, w, label=rem)
	ax.set_xticks(pos)
	ax.set_xticklabels(pv.index)
	ax.axhline(0, color="black", lw=0.6)
	ax.set_ylabel("predicted drop in 2-hour peak (mg/dL)")
	ax.set_title("What-if remedies on unseen patients (model prediction)")
	ax.legend(fontsize=8)
	fig.tight_layout()
	fig.savefig("plots/remedies.png", dpi=130)
	print("\nSaved: models/remedy_results_step5.csv, plots/remedies.png")


if __name__ == "__main__":
	main()
