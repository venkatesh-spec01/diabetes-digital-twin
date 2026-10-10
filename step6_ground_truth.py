"""
STEP 6 - Ground-truth check of the what-if remedies.

For every real meal of the 10 unseen test patients, we compute:
  - the TRUE drop in 2-hour peak glucose, using the simulator's own meal formula
  - the PREDICTED drop, from the fusion model (same as step 5)
and compare them.

Run:  python step6_ground_truth.py
"""
import numpy as np
import pandas as pd
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from step1_generate_data import kernel
from step2_train_model import DATA, HISTORY, HORIZON, load_patient
from step3_fusion import load_ehr, FusionForecaster
from step5_remedies import make_variant, forecast, AFTER, REMEDIES, START


def meal_curve(p, carbs, gi, walked, n=72):
	"""Glucose change caused by one meal (rise minus medicine effect), same formula as step 1."""
	seg = np.arange(n) * 5.0
	ir = p["insulin_resistance"] * (0.85 if p["metformin"] else 1.0)  # sleep-deficit term ignored
	amp = 0.6 * carbs * (gi / 70) * ir * (0.72 if walked else 1.0)
	med = p["hypo_strength"] * 0.85 * carbs * (gi / 70)
	return amp * kernel(seg, 45) - med * kernel(seg, 150)


def remedy_params(remedy, carbs, gi):
	if remedy == "walk 10 min":
		return carbs, gi, 1
	if remedy == "30% smaller portion":
		return carbs * 0.7, gi, 0
	if remedy == "low-GI swap (GI 50)":
		return (carbs, 50, 0) if gi > 50 else None
	return carbs * 0.7, gi, 1  # walk + smaller portion


def main():
	patients = pd.read_csv(f"{DATA}/patients.csv")
	train_ids = patients[patients.group == "train"].patient_id.tolist()
	test_ids = patients[patients.group == "test"].patient_id.tolist()
	ehr = load_ehr(patients, train_ids)
	meals = pd.read_csv(f"{DATA}/meals.csv", parse_dates=["timestamp"])
	pat = patients.set_index("patient_id")

	model = FusionForecaster(10, ehr.shape[1])
	model.load_state_dict(torch.load("models/fusion_lstm.pt"))
	model.eval()

	rows = []
	for pid in test_ids:
		x, g = load_patient(pid)
		s = ehr.loc[pid].values.astype(np.float32)
		p = pat.loc[pid].to_dict()
		m = meals[(meals.patient_id == pid) & (meals.walked_after == 0)]
		info, base_W = [], []
		for _, r in m.iterrows():
			t0 = int(round((r.timestamp - START).total_seconds() / 300))
			i = t0 + AFTER
			if t0 < 1 or i < HISTORY or i + HORIZON > len(g):
				continue
			info.append((t0, float(r.carbs_g), float(r.gi)))
			base_W.append(x[i - HISTORY:i].copy())
		if not base_W:
			continue
		peak0 = forecast(model, np.array(base_W), s).max(axis=1)
		for remedy in REMEDIES:
			W, keep = [], []
			for k, (t0, carbs, gi) in enumerate(info):
				w = make_variant(x, t0, gi, remedy)
				if w is not None:
					W.append(w)
					keep.append(k)
			if not W:
				continue
			peak1 = forecast(model, np.array(W), s).max(axis=1)
			for j, k in enumerate(keep):
				t0, carbs, gi = info[k]
				i = t0 + AFTER
				c2, gi2, walked2 = remedy_params(remedy, carbs, gi)
				d = meal_curve(p, c2, gi2, walked2) - meal_curve(p, carbs, gi, 0)
				real = g[i:i + HORIZON]
				true_drop = real.max() - (real + d[AFTER:AFTER + HORIZON]).max()
				rows.append(dict(patient_id=pid, remedy=remedy,
								 pred_drop=peak0[k] - peak1[j], true_drop=true_drop))

	df = pd.DataFrame(rows)
	df["error"] = df.pred_drop - df.true_drop
	df.to_csv("models/remedy_truth_step6.csv", index=False)

	out = []
	for rem in REMEDIES:
		d = df[df.remedy == rem]
		out.append(dict(remedy=rem, meals=len(d), true_avg=d.true_drop.mean(),
						predicted_avg=d.pred_drop.mean(), bias=d.error.mean(),
						mae=d.error.abs().mean(),
						correlation=np.corrcoef(d.pred_drop, d.true_drop)[0, 1]))
	res = pd.DataFrame(out).set_index("remedy").round(2)
	print("GROUND TRUTH vs MODEL (drop in 2-hour peak, mg/dL)")
	print(res.to_string())
	print("\nbias = predicted minus true (negative = model underestimates the remedy)")
	print("mae = typical error per meal; correlation = does the model rank meals correctly (1 = perfect)")

	fig, axes = plt.subplots(1, 4, figsize=(15, 3.8))
	for ax, rem in zip(axes, REMEDIES):
		d = df[df.remedy == rem]
		ax.scatter(d.true_drop, d.pred_drop, s=6, alpha=0.5)
		lim = max(d.true_drop.max(), d.pred_drop.max()) + 2
		ax.plot([0, lim], [0, lim], color="red", lw=0.8)
		ax.set_title(rem, fontsize=9)
		ax.set_xlabel("true drop")
		ax.set_ylabel("predicted drop")
	fig.tight_layout()
	fig.savefig("plots/remedy_truth.png", dpi=130)
	print("\nSaved: models/remedy_truth_step6.csv, plots/remedy_truth.png")


if __name__ == "__main__":
	main()
