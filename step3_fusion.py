"""
STEP 3 - Fusion model: sensor history + health record (EHR).

What it does:
  1. Reads each patient's health record (age, HbA1c, medicines, ...) from data/patients.csv.
  2. Trains an LSTM that reads the last 6 hours of sensor data AND the health record,
	 then predicts the next 2 hours of glucose.
  3. Compares it with the glucose-only model from step 2 on the 10 unseen test patients.

Run:  python step3_fusion.py
"""
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from step2_train_model import (DATA, HORIZON, BATCH, LR, EPOCHS, SEED,
							   load_patient, make_windows, rmse_by_horizon, Forecaster)
from step2_train_model import predict as predict_seq

EHR_COLUMNS = ["age", "sex", "bmi", "years_since_diagnosis", "hba1c", "fasting_glucose",
			   "ldl", "triglycerides", "family_history", "polygenic_risk",
			   "metformin", "sulfonylurea", "insulin"]


def load_ehr(patients, train_ids):
	"""One row of numbers per patient, scaled using the TRAIN patients only."""
	ehr = patients.set_index("patient_id")[EHR_COLUMNS].copy()
	ehr = pd.get_dummies(ehr, dtype=float)
	ehr = ehr.astype(float).fillna(ehr.median())
	mu = ehr.loc[train_ids].mean()
	sd = ehr.loc[train_ids].std().replace(0, 1).fillna(1)
	return (ehr - mu) / sd


def build_fused(ids, ehr, split_rows=None):
	Xs, Ss, Ys = [], [], []
	for pid in ids:
		x, g = load_patient(pid)
		a, b = (0, len(g)) if split_rows is None else split_rows
		X, Y, _ = make_windows(x, g, a, b)
		s = ehr.loc[pid].values.astype(np.float32)
		Xs.append(X)
		Ys.append(Y)
		Ss.append(np.repeat(s[None], len(X), axis=0))
	return np.concatenate(Xs), np.concatenate(Ss), np.concatenate(Ys)


class FusionForecaster(nn.Module):
	def __init__(self, n_seq, n_static, hidden=64):
		super().__init__()
		self.lstm = nn.LSTM(n_seq, hidden, batch_first=True)
		self.ehr = nn.Sequential(nn.Linear(n_static, 32), nn.ReLU(),
								 nn.Linear(32, 32), nn.ReLU())
		self.head = nn.Sequential(nn.Linear(hidden + 32, 64), nn.ReLU(),
								  nn.Linear(64, HORIZON))

	def forward(self, x, s):
		out, _ = self.lstm(x)
		return self.head(torch.cat([out[:, -1], self.ehr(s)], dim=1))


def predict_fused(model, X, S):
	model.eval()
	out = []
	with torch.no_grad():
		for i in range(0, len(X), 2048):
			out.append(model(torch.from_numpy(X[i:i + 2048]),
							 torch.from_numpy(S[i:i + 2048])).numpy())
	return np.concatenate(out)


def main():
	torch.manual_seed(SEED)
	np.random.seed(SEED)
	patients = pd.read_csv(f"{DATA}/patients.csv")
	train_ids = patients[patients.group == "train"].patient_id.tolist()
	test_ids = patients[patients.group == "test"].patient_id.tolist()
	ehr = load_ehr(patients, train_ids)

	n_rows = len(pd.read_csv(f"{DATA}/sensors/{train_ids[0]}.csv"))
	cut = int(n_rows * 0.8)
	Xtr, Str, Ytr = build_fused(train_ids, ehr, (0, cut))
	Xva, Sva, Yva = build_fused(train_ids, ehr, (cut, n_rows))
	print(f"Training windows: {len(Xtr)} | validation windows: {len(Xva)} | "
		  f"sensor inputs: {Xtr.shape[2]} | health-record inputs: {Str.shape[1]}")

	model = FusionForecaster(Xtr.shape[2], Str.shape[1])
	opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
	loss_fn = nn.MSELoss()
	Xt, St, Yt = torch.from_numpy(Xtr), torch.from_numpy(Str), torch.from_numpy(Ytr)

	best, t0 = 1e9, time.time()
	for ep in range(1, EPOCHS + 1):
		model.train()
		perm = torch.randperm(len(Xt))
		total = 0.0
		for i in range(0, len(Xt), BATCH):
			idx = perm[i:i + BATCH]
			s_batch = St[idx] + 0.1 * torch.randn_like(St[idx])
			opt.zero_grad()
			loss = loss_fn(model(Xt[idx], s_batch), Yt[idx])
			loss.backward()
			nn.utils.clip_grad_norm_(model.parameters(), 1.0)
			opt.step()
			total += loss.item() * len(idx)
		va = rmse_by_horizon(predict_fused(model, Xva, Sva), Yva)
		print(f"epoch {ep:2d}/{EPOCHS}  train loss {total / len(Xt):.4f}  "
			  f"validation RMSE 30/60/120 min: {va['30min']:.1f} / {va['60min']:.1f} / {va['120min']:.1f} mg/dL  "
			  f"({time.time() - t0:.0f}s)")
		if va["120min"] < best:
			best = va["120min"]
			torch.save(model.state_dict(), "models/fusion_lstm.pt")
	model.load_state_dict(torch.load("models/fusion_lstm.pt"))

	# Compare against the glucose-only model from step 2.
	base = Forecaster(Xtr.shape[2])
	base.load_state_dict(torch.load("models/baseline_lstm.pt"))

	lazy = rmse_by_horizon(np.zeros_like(Yva), Yva)
	b = rmse_by_horizon(predict_seq(base, Xva), Yva)
	f = rmse_by_horizon(predict_fused(model, Xva, Sva), Yva)
	print("\nVALIDATION (training patients, later days)   RMSE 30/60/120 min")
	print(f"  lazy guess        : {lazy['30min']:.1f} / {lazy['60min']:.1f} / {lazy['120min']:.1f}")
	print(f"  glucose-only LSTM : {b['30min']:.1f} / {b['60min']:.1f} / {b['120min']:.1f}")
	print(f"  fusion (+ health record): {f['30min']:.1f} / {f['60min']:.1f} / {f['120min']:.1f}")

	print("\nHARD TEST PATIENTS (never seen)   RMSE in mg/dL, smaller is better")
	print("  patient  lazy@120  glucose-only@120  fusion@120")
	rows = []
	for pid in test_ids:
		X, S, Y = build_fused([pid], ehr)
		l = rmse_by_horizon(np.zeros_like(Y), Y)
		bb = rmse_by_horizon(predict_seq(base, X), Y)
		ff = rmse_by_horizon(predict_fused(model, X, S), Y)
		print(f"  {pid}     {l['120min']:7.1f}   {bb['120min']:10.1f}      {ff['120min']:9.1f}")
		rows.append(dict(patient_id=pid, lazy_120=l["120min"], glucose_only_120=bb["120min"],
						 fusion_120=ff["120min"], glucose_only_60=bb["60min"], fusion_60=ff["60min"]))
	df = pd.DataFrame(rows)
	df.to_csv("models/test_results_step3.csv", index=False)

	wins = int((df.fusion_120 < df.glucose_only_120).sum())
	print(f"\nAverage 2-hour error on unseen patients: glucose-only {df.glucose_only_120.mean():.1f}"
		  f"  vs  fusion {df.fusion_120.mean():.1f} mg/dL")
	print(f"Fusion beats glucose-only for {wins} of {len(df)} unseen patients")

	fig, ax = plt.subplots(figsize=(10, 4))
	pos = np.arange(len(df))
	ax.bar(pos - 0.2, df.glucose_only_120, 0.4, label="glucose only", color="gray")
	ax.bar(pos + 0.2, df.fusion_120, 0.4, label="glucose + health record", color="tab:green")
	ax.set_xticks(pos)
	ax.set_xticklabels(df.patient_id)
	ax.set_ylabel("2-hour RMSE (mg/dL), lower is better")
	ax.set_title("Unseen patients: does the health record help?")
	ax.legend()
	fig.tight_layout()
	fig.savefig("plots/fusion_vs_baseline.png", dpi=130)
	print("Saved: models/fusion_lstm.pt, models/test_results_step3.csv, plots/fusion_vs_baseline.png")


if __name__ == "__main__":
	main()
