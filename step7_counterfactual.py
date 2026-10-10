"""
STEP 7 - Counterfactual fine-tuning of the fusion model.

Teaches the model what a remedy (walk, smaller portion, low-GI swap) really does,
using "same meal with / without the remedy" pairs from the simulator formula.
Then re-runs the ground-truth check on the 10 unseen test patients (old vs new model).

Run:  python step7_counterfactual.py
"""
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from step2_train_model import DATA, HISTORY, HORIZON, BATCH, SEED, load_patient, rmse_by_horizon
from step3_fusion import load_ehr, build_fused, FusionForecaster, predict_fused
from step5_remedies import make_variant, forecast, AFTER, REMEDIES, START
from step6_ground_truth import meal_curve, remedy_params

FT_EPOCHS = 6
FT_LR = 5e-4
REPEAT_CF = 2          # how many times the counterfactual examples are shown per round


def cf_samples(ids, ehr, pat, meals, lo, hi):
	"""Counterfactual training examples from rows [lo, hi) of each patient's timeline."""
	Xs, Ss, Ys = [], [], []
	for pid in ids:
		x, g = load_patient(pid)
		s = ehr.loc[pid].values.astype(np.float32)
		p = pat.loc[pid].to_dict()
		m = meals[(meals.patient_id == pid) & (meals.walked_after == 0)]
		for _, r in m.iterrows():
			t0 = int(round((r.timestamp - START).total_seconds() / 300))
			i = t0 + AFTER
			carbs, gi = float(r.carbs_g), float(r.gi)
			if t0 < 1 or i - HISTORY < lo or i + HORIZON > hi:
				continue
			for remedy in REMEDIES:
				w = make_variant(x, t0, gi, remedy)
				if w is None:
					continue
				c2, gi2, walked2 = remedy_params(remedy, carbs, gi)
				d = meal_curve(p, c2, gi2, walked2) - meal_curve(p, carbs, gi, 0)
				target = (g[i:i + HORIZON] + d[AFTER:AFTER + HORIZON] - g[i - 1]) / 100.0
				Xs.append(w.astype(np.float32))
				Ss.append(s)
				Ys.append(target.astype(np.float32))
	return np.array(Xs), np.array(Ss), np.array(Ys)


def truth_table(model, test_ids, ehr, pat, meals):
	"""Predicted vs true drop in 2-hour peak for each remedy (same method as step 6)."""
	rows = []
	for pid in test_ids:
		x, g = load_patient(pid)
		s = ehr.loc[pid].values.astype(np.float32)
		p = pat.loc[pid].to_dict()
		m = meals[(meals.patient_id == pid) & (meals.walked_after == 0)]
		info, base = [], []
		for _, r in m.iterrows():
			t0 = int(round((r.timestamp - START).total_seconds() / 300))
			i = t0 + AFTER
			if t0 < 1 or i < HISTORY or i + HORIZON > len(g):
				continue
			info.append((t0, float(r.carbs_g), float(r.gi)))
			base.append(x[i - HISTORY:i].copy())
		peak0 = forecast(model, np.array(base), s).max(axis=1)
		for remedy in REMEDIES:
			W, keep = [], []
			for k, (t0, carbs, gi) in enumerate(info):
				w = make_variant(x, t0, gi, remedy)
				if w is not None:
					W.append(w)
					keep.append(k)
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
	out = []
	for rem in REMEDIES:
		d = df[df.remedy == rem]
		out.append(dict(remedy=rem, true_avg=d.true_drop.mean(), predicted_avg=d.pred_drop.mean(),
						bias=d.error.mean(), mae=d.error.abs().mean(),
						correlation=np.corrcoef(d.pred_drop, d.true_drop)[0, 1]))
	return pd.DataFrame(out).set_index("remedy").round(2)


def test_forecast_error(model, test_ids, ehr):
	errs = []
	for pid in test_ids:
		X, S, Y = build_fused([pid], ehr)
		errs.append(rmse_by_horizon(predict_fused(model, X, S), Y)["120min"])
	return np.array(errs)


def main():
	torch.manual_seed(SEED)
	np.random.seed(SEED)
	patients = pd.read_csv(f"{DATA}/patients.csv")
	train_ids = patients[patients.group == "train"].patient_id.tolist()
	test_ids = patients[patients.group == "test"].patient_id.tolist()
	ehr = load_ehr(patients, train_ids)
	pat = patients.set_index("patient_id")
	meals = pd.read_csv(f"{DATA}/meals.csv", parse_dates=["timestamp"])
	n_rows = len(pd.read_csv(f"{DATA}/sensors/{train_ids[0]}.csv"))
	cut = int(n_rows * 0.8)

	Xtr, Str, Ytr = build_fused(train_ids, ehr, (0, cut))
	Xva, Sva, Yva = build_fused(train_ids, ehr, (cut, n_rows))
	Xc, Sc, Yc = cf_samples(train_ids, ehr, pat, meals, 0, cut)
	Xcv, Scv, Ycv = cf_samples(train_ids, ehr, pat, meals, cut, n_rows)
	print(f"Original windows: {len(Xtr)} | counterfactual examples: {len(Xc)} "
		  f"(validation: {len(Xcv)})")

	X = torch.from_numpy(np.concatenate([Xtr] + [Xc] * REPEAT_CF))
	S = torch.from_numpy(np.concatenate([Str] + [Sc] * REPEAT_CF))
	Y = torch.from_numpy(np.concatenate([Ytr] + [Yc] * REPEAT_CF))
	del Xtr, Str, Ytr

	model = FusionForecaster(9, S.shape[1])
	model.load_state_dict(torch.load("models/fusion_lstm.pt"))
	opt = torch.optim.Adam(model.parameters(), lr=FT_LR, weight_decay=1e-5)
	loss_fn = nn.MSELoss()

	best, t0 = 1e9, time.time()
	for ep in range(1, FT_EPOCHS + 1):
		model.train()
		perm = torch.randperm(len(X))
		total = 0.0
		for i in range(0, len(X), BATCH):
			idx = perm[i:i + BATCH]
			s_batch = S[idx] + 0.1 * torch.randn_like(S[idx])
			opt.zero_grad()
			loss = loss_fn(model(X[idx], s_batch), Y[idx])
			loss.backward()
			nn.utils.clip_grad_norm_(model.parameters(), 1.0)
			opt.step()
			total += loss.item() * len(idx)
		a = rmse_by_horizon(predict_fused(model, Xva, Sva), Yva)["120min"]
		b = rmse_by_horizon(predict_fused(model, Xcv, Scv), Ycv)["120min"]
		print(f"round {ep}/{FT_EPOCHS}  train loss {total / len(X):.4f}  "
			  f"validation 2h RMSE: normal windows {a:.1f} | remedy examples {b:.1f} mg/dL  "
			  f"({time.time() - t0:.0f}s)")
		if a + b < best:
			best = a + b
			torch.save(model.state_dict(), "models/fusion_cf.pt")
	model.load_state_dict(torch.load("models/fusion_cf.pt"))

	old = FusionForecaster(9, S.shape[1])
	old.load_state_dict(torch.load("models/fusion_lstm.pt"))

	print("\nGROUND TRUTH on 10 unseen patients (drop in 2-hour peak, mg/dL)")
	t_old = truth_table(old, test_ids, ehr, pat, meals)
	t_new = truth_table(model, test_ids, ehr, pat, meals)
	print("\nOLD model (step 3):")
	print(t_old.to_string())
	print("\nNEW model (counterfactual fine-tuned):")
	print(t_new.to_string())
	print("\nbias = predicted minus true (closer to 0 is better); mae = typical error per meal")

	e_old = test_forecast_error(old, test_ids, ehr)
	e_new = test_forecast_error(model, test_ids, ehr)
	print(f"\nNORMAL 2-hour forecast error on unseen patients (average): "
		  f"old {e_old.mean():.1f}  vs  new {e_new.mean():.1f} mg/dL")
	print(f"New model is better than old for {int((e_new < e_old).sum())} of {len(test_ids)} patients")

	t_old["model"] = "old"
	t_new["model"] = "new"
	pd.concat([t_old, t_new]).to_csv("models/remedy_truth_step7.csv")
	print("\nSaved: models/fusion_cf.pt, models/remedy_truth_step7.csv")


if __name__ == "__main__":
	main()
