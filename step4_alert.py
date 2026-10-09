"""
STEP 4 - Adverse event alerts (high and low glucose in the next 2 hours).

Turns the 2-hour forecast into alerts and measures them on the 10 unseen test patients.
Only NEW events are counted (glucose was in range when the alert was raised).

Run:  python step4_alerts.py
"""
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from step2_train_model import (DATA, HORIZON, load_patient, make_windows, Forecaster,
							   predict as predict_seq)
from step3_fusion import load_ehr, FusionForecaster, predict_fused

HIGH, LOW = 180.0, 70.0


def collect(ids, ehr):
	Xs, Ss, Fs, Ls, Ps = [], [], [], [], []
	for pid in ids:
		x, g = load_patient(pid)
		X, Y, L = make_windows(x, g, 0, len(g))
		s = ehr.loc[pid].values.astype(np.float32)
		Xs.append(X)
		Ss.append(np.repeat(s[None], len(X), axis=0))
		Fs.append(L[:, None] + Y * 100.0)  # real glucose over the next 2 hours
		Ls.append(L)
		Ps.extend([pid] * len(X))
	return (np.concatenate(Xs), np.concatenate(Ss), np.concatenate(Fs),
			np.concatenate(Ls), np.array(Ps))


def score(alert, event):
	tp = int((alert & event).sum())
	fp = int((alert & ~event).sum())
	fn = int((~alert & event).sum())
	precision = tp / (tp + fp) if tp + fp else 0.0
	recall = tp / (tp + fn) if tp + fn else 0.0
	f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
	return precision, recall, f1, tp, fp, fn


def main():
	patients = pd.read_csv(f"{DATA}/patients.csv")
	train_ids = patients[patients.group == "train"].patient_id.tolist()
	test_ids = patients[patients.group == "test"].patient_id.tolist()
	ehr = load_ehr(patients, train_ids)
	X, S, future, now, _ = collect(test_ids, ehr)
	print(f"Test windows: {len(X)} from {len(test_ids)} unseen patients")

	base = Forecaster(X.shape[2])
	base.load_state_dict(torch.load("models/baseline_lstm.pt"))
	fuse = FusionForecaster(X.shape[2], S.shape[1])
	fuse.load_state_dict(torch.load("models/fusion_lstm.pt"))

	steps = np.arange(1, HORIZON + 1)
	glucose_hist = X[:, :, 0] * 200.0
	slope = (glucose_hist[:, -1] - glucose_hist[:, -7]) / 6.0
	forecasts = {
		"trend baseline": now[:, None] + slope[:, None] * steps,
		"glucose-only LSTM": now[:, None] + predict_seq(base, X) * 100.0,
		"fusion LSTM": now[:, None] + predict_fused(fuse, X, S) * 100.0,
	}
	methods = [
		("trend baseline", "trend baseline", 0),
		("glucose-only LSTM", "glucose-only LSTM", 0),
		("fusion LSTM", "fusion LSTM", 0),
		("fusion LSTM, sensitive", "fusion LSTM", 15),
	]

	rows = []
	for kind in ("HIGH (>180)", "LOW (<70)"):
		if kind.startswith("HIGH"):
			mask = now <= HIGH
			event = future.max(axis=1) > HIGH
		else:
			mask = now >= LOW
			event = future.min(axis=1) < LOW
		print(f"\n{kind}: {int(event[mask].sum())} new events among {int(mask.sum())} windows")
		print("  method                     precision  recall   F1    caught  false alarms  missed")
		for name, key, margin in methods:
			forecast = forecasts[key]
			if kind.startswith("HIGH"):
				alert = forecast.max(axis=1) > HIGH - margin
			else:
				alert = forecast.min(axis=1) < LOW + margin
			p, r, f1, tp, fp, fn = score(alert[mask], event[mask])
			print(f"  {name:25s}  {p:7.2f}   {r:6.2f}  {f1:5.2f}  {tp:6d}  {fp:12d}  {fn:6d}")
			rows.append(dict(event=kind, method=name, precision=p, recall=r, f1=f1,
							 caught=tp, false_alarms=fp, missed=fn))

	df = pd.DataFrame(rows)
	df.to_csv("models/alert_results_step4.csv", index=False)

	fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
	for ax, kind in zip(axes, ("HIGH (>180)", "LOW (<70)")):
		data = df[df.event == kind]
		ax.bar(data.method, data.f1, color=["gray", "silver", "tab:green", "tab:olive"])
		ax.set_title(f"{kind} alert quality (F1, higher is better)")
		ax.tick_params(axis="x", rotation=25)
	fig.tight_layout()
	fig.savefig("plots/alert_quality.png", dpi=130)
	print("\nSaved: models/alert_results_step4.csv, plots/alert_quality.png")


if __name__ == "__main__":
	main()
