"""
STEP 8 - Is fusion really better than glucose-only? Train both with several random seeds.

Does NOT touch the final models. Run:  python step8_seeds.py
"""
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from step2_train_model import (DATA, BATCH, LR, EPOCHS, rmse_by_horizon,
							   Forecaster, predict as predict_seq)
from step3_fusion import load_ehr, build_fused, FusionForecaster, predict_fused

SEEDS = [1, 2, 3, 4, 5]


def train(model, fused, Xtr, Str, Ytr, Xva, Sva, Yva, seed):
	torch.manual_seed(seed)
	np.random.seed(seed)
	opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5 if fused else 0.0)
	loss_fn = nn.MSELoss()
	Xt, Yt = torch.from_numpy(Xtr), torch.from_numpy(Ytr)
	St = torch.from_numpy(Str)
	best, best_state = 1e9, None
	for ep in range(EPOCHS):
		model.train()
		perm = torch.randperm(len(Xt))
		for i in range(0, len(Xt), BATCH):
			idx = perm[i:i + BATCH]
			opt.zero_grad()
			if fused:
				out = model(Xt[idx], St[idx] + 0.1 * torch.randn_like(St[idx]))
			else:
				out = model(Xt[idx])
			loss_fn(out, Yt[idx]).backward()
			nn.utils.clip_grad_norm_(model.parameters(), 1.0)
			opt.step()
		p = predict_fused(model, Xva, Sva) if fused else predict_seq(model, Xva)
		v = rmse_by_horizon(p, Yva)["120min"]
		if v < best:
			best = v
			best_state = {k: t.clone() for k, t in model.state_dict().items()}
	model.load_state_dict(best_state)
	return model


def main():
	patients = pd.read_csv(f"{DATA}/patients.csv")
	train_ids = patients[patients.group == "train"].patient_id.tolist()
	test_ids = patients[patients.group == "test"].patient_id.tolist()
	ehr = load_ehr(patients, train_ids)
	n_rows = len(pd.read_csv(f"{DATA}/sensors/{train_ids[0]}.csv"))
	cut = int(n_rows * 0.8)

	Xtr, Str, Ytr = build_fused(train_ids, ehr, (0, cut))
	Xva, Sva, Yva = build_fused(train_ids, ehr, (cut, n_rows))
	tests = {pid: build_fused([pid], ehr) for pid in test_ids}
	print(f"Training windows: {len(Xtr)} | seeds: {SEEDS}\n")

	rows = []
	t0 = time.time()
	for seed in SEEDS:
		torch.manual_seed(seed)
		base = train(Forecaster(Xtr.shape[2]), False, Xtr, Str, Ytr, Xva, Sva, Yva, seed)
		torch.manual_seed(seed)
		fuse = train(FusionForecaster(Xtr.shape[2], Str.shape[1]), True,
					 Xtr, Str, Ytr, Xva, Sva, Yva, seed)
		eb, ef = [], []
		for pid in test_ids:
			X, S, Y = tests[pid]
			eb.append(rmse_by_horizon(predict_seq(base, X), Y)["120min"])
			ef.append(rmse_by_horizon(predict_fused(fuse, X, S), Y)["120min"])
		eb, ef = np.array(eb), np.array(ef)
		rows.append(dict(seed=seed, glucose_only=eb.mean(), fusion=ef.mean(),
						 fusion_wins=int((ef < eb).sum())))
		print(f"seed {seed}: glucose-only {eb.mean():.2f} | fusion {ef.mean():.2f} | "
			  f"fusion wins {int((ef < eb).sum())} of {len(test_ids)} patients  ({time.time() - t0:.0f}s)")

	df = pd.DataFrame(rows)
	df.to_csv("models/seed_results_step8.csv", index=False)
	diff = df.glucose_only - df.fusion
	print("\n2-hour error on 10 unseen patients (mg/dL, lower is better)")
	print(f"  glucose-only : {df.glucose_only.mean():.2f} +/- {df.glucose_only.std():.2f}")
	print(f"  fusion       : {df.fusion.mean():.2f} +/- {df.fusion.std():.2f}")
	print(f"  fusion better in {int((diff > 0).sum())} of {len(df)} seeds; "
		  f"average gain {diff.mean():.2f} mg/dL (spread {diff.std():.2f})")
	print("\nSaved: models/seed_results_step8.csv")


if __name__ == "__main__":
	main()
