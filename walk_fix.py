# fixes the walk remedy bias.
# after whatif_finetune the model promised too much for walking, and before
# it promised too little, so the right amount is somewhere in between.
# here we check the remedy bias after every round on validation meals
# (later days of the training patients) and keep the best round.
# then we try blends of old and new weights, again on validation only.
# the test patients are used only once at the end.
# saves models/fusion_cf_v2.pt and models/walk_fix_truth.csv
# run: python walk_fix.py
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from train_baseline import DATA, HISTORY, HORIZON, BATCH, SEED, load_patient, rmse_by_horizon
from fusion_model import load_ehr, build_fused, FusionForecaster, predict_fused
from whatif import make_variant, forecast, AFTER, REMEDIES, START
from whatif_truth_check import meal_curve, remedy_params
from whatif_finetune import (cf_samples, truth_table, test_forecast_error,
								  FT_EPOCHS, FT_LR, REPEAT_CF)

# how much of the new weights to mix
BLENDS = [0.0, 0.25, 0.5, 0.75, 1.0]


# average predicted-minus-true error for each remedy
def remedy_bias(model, ids, ehr, pat, meals, lo, hi):
	rows = []
	for pid in ids:
		x, g = load_patient(pid)
		s = ehr.loc[pid].values.astype(np.float32)
		p = pat.loc[pid].to_dict()
		m = meals[(meals.patient_id == pid) & (meals.walked_after == 0)]
		info, base = [], []
		for _, r in m.iterrows():
			t0 = int(round((r.timestamp - START).total_seconds() / 300))
			i = t0 + AFTER
			if t0 < 1 or i - HISTORY < lo or i + HORIZON > hi:
				continue
			info.append((t0, float(r.carbs_g), float(r.gi)))
			base.append(x[i - HISTORY:i].copy())
		if not base:
			continue
		peak0 = forecast(model, np.array(base), s).max(axis=1)
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
				rows.append(dict(remedy=remedy, error=(peak0[k] - peak1[j]) - true_drop))
	df = pd.DataFrame(rows)
	return df.groupby("remedy").error.mean().reindex(REMEDIES)


# combined score on validation, lower is better
def val_score(model, Xva, Sva, Yva, Xcv, Scv, Ycv, train_ids, ehr, pat, meals, cut, n_rows):
	a = rmse_by_horizon(predict_fused(model, Xva, Sva), Yva)["120min"]
	b = rmse_by_horizon(predict_fused(model, Xcv, Scv), Ycv)["120min"]
	bias = remedy_bias(model, train_ids, ehr, pat, meals, cut, n_rows)
	return a + b + bias.abs().mean(), a, b, bias


# prints one line of results
def show(tag, score, a, b, bias):
	short = ["walk", "portion", "lowGI", "walk+portion"]
	parts = "  ".join(f"{n} {v:+.2f}" for n, v in zip(short, bias.values))
	print(f"{tag:<14} score {score:6.2f} | RMSE normal {a:5.1f} remedy {b:5.1f} | bias: {parts}")


# fine-tunes, tries blends, picks best on validation
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
	X = torch.from_numpy(np.concatenate([Xtr] + [Xc] * REPEAT_CF))
	S = torch.from_numpy(np.concatenate([Str] + [Sc] * REPEAT_CF))
	Y = torch.from_numpy(np.concatenate([Ytr] + [Yc] * REPEAT_CF))
	del Xtr, Str, Ytr
	args = (Xva, Sva, Yva, Xcv, Scv, Ycv, train_ids, ehr, pat, meals, cut, n_rows)

	old = FusionForecaster(10, S.shape[1])
	old.load_state_dict(torch.load("models/fusion_lstm.pt"))
	print("VALIDATION only (later days of training patients). Lower score is better.\n")
	# score of the old model first
	best = val_score(old, *args)
	show("old model", *best)
	best_state = {k: v.clone() for k, v in old.state_dict().items()}

	model = FusionForecaster(10, S.shape[1])
	model.load_state_dict(torch.load("models/fusion_lstm.pt"))
	opt = torch.optim.Adam(model.parameters(), lr=FT_LR, weight_decay=1e-5)
	loss_fn = nn.MSELoss()
	t0 = time.time()
	for ep in range(1, FT_EPOCHS + 1):
		model.train()
		perm = torch.randperm(len(X))
		for i in range(0, len(X), BATCH):
			idx = perm[i:i + BATCH]
			s_batch = S[idx] + 0.1 * torch.randn_like(S[idx])
			opt.zero_grad()
			loss = loss_fn(model(X[idx], s_batch), Y[idx])
			loss.backward()
			nn.utils.clip_grad_norm_(model.parameters(), 1.0)
			opt.step()
		res = val_score(model, *args)
		show(f"round {ep}/{FT_EPOCHS}", *res)
		print(f"{'':<14} ({time.time() - t0:.0f}s)")
		if res[0] < best[0]:
			best = res
			best_state = {k: v.clone() for k, v in model.state_dict().items()}

	print("\nBlending old and fine-tuned weights (validation only):")
	old_state = old.state_dict()
	final_state, final = best_state, best
	for a in BLENDS:
		# mix old and new weights by percentage
		st = {k: (1 - a) * old_state[k] + a * best_state[k] for k in old_state}
		m = FusionForecaster(10, S.shape[1])
		m.load_state_dict(st)
		res = val_score(m, *args)
		show(f"blend {int(a * 100)}%", *res)
		# keep the blend with the best score
		if res[0] < final[0]:
			final, final_state = res, st
	torch.save(final_state, "models/fusion_cf_v2.pt")
	print(f"\nChosen on validation: score {final[0]:.2f}. Saved models/fusion_cf_v2.pt")

	new = FusionForecaster(10, S.shape[1])
	new.load_state_dict(final_state)
	step7 = FusionForecaster(10, S.shape[1])
	step7.load_state_dict(torch.load("models/fusion_cf.pt"))
	tables = []
	# finally test all three on test patients
	for tag, m in [("old", old), ("step7", step7), ("step9", new)]:
		t = truth_table(m, test_ids, ehr, pat, meals)
		e = test_forecast_error(m, test_ids, ehr)
		print(f"\nTEST, {tag} model   (2-hour forecast RMSE {e.mean():.2f} mg/dL)")
		print(t.to_string())
		t["model"] = tag
		t["test_rmse_2h"] = round(e.mean(), 2)
		tables.append(t)
	pd.concat(tables).to_csv("models/walk_fix_truth.csv")
	print("\nSaved: models/walk_fix_truth.csv")
	print("If step9 has smaller walk biases and similar RMSE, switch the dashboard to fusion_cf_v2.pt.")


if __name__ == "__main__":
	main()
