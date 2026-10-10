"""
Doctor dashboard for the Type 2 Diabetes digital twin (synthetic data only).

Run:  streamlit run app.py
"""
import numpy as np
import pandas as pd
import torch
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from step2_train_model import DATA, HISTORY, HORIZON, load_patient
from step3_fusion import load_ehr, FusionForecaster
from step5_remedies import make_variant, forecast, AFTER, REMEDIES, START

HIGH, LOW = 180.0, 70.0
MARGIN_LOW = 15.0     # more sensitive setting for lows (a missed low is the dangerous mistake)

st.set_page_config(page_title="T2D Digital Twin", layout="wide")


@st.cache_resource
def load_everything():
	patients = pd.read_csv(f"{DATA}/patients.csv")
	train_ids = patients[patients.group == "train"].patient_id.tolist()
	ehr = load_ehr(patients, train_ids)
	meals = pd.read_csv(f"{DATA}/meals.csv", parse_dates=["timestamp"])
	model = FusionForecaster(10, ehr.shape[1])
	model.load_state_dict(torch.load("models/fusion_cf.pt"))
	model.eval()
	return patients, ehr, meals, model


patients, ehr, meals, model = load_everything()
test_ids = patients[patients.group == "test"].patient_id.tolist()

st.title("Type 2 Diabetes Digital Twin")
st.caption("Prototype on 100% synthetic data. Not medical advice. Needs clinical validation.")

pid = st.sidebar.selectbox("Patient (unseen during training)", test_ids)
row = patients[patients.patient_id == pid].iloc[0]
x, g = load_patient(pid)
s = ehr.loc[pid].values.astype(np.float32)

n_steps = len(g)
default_t = 3 * 288 + 150
t = st.sidebar.slider("Moment in time (5-minute steps)", HISTORY, n_steps - HORIZON - 1, default_t)
st.sidebar.write(f"Time: {(START + pd.Timedelta(minutes=5 * t)).strftime('%d %b %Y, %H:%M')}")

# ---------------- left: health record ----------------
left, right = st.columns([1, 2])
with left:
	st.subheader(f"Patient {pid}")
	st.write(f"**Age** {row.age} | **Sex** {row.sex} | **BMI** {row.bmi}")
	st.write(f"**Years since diagnosis** {row.years_since_diagnosis}")
	st.write(f"**HbA1c** {row.hba1c}% | **Fasting glucose** {row.fasting_glucose} mg/dL")
	st.write(f"**LDL** {row.ldl} | **Triglycerides** {row.triglycerides}")
	meds = [n for n, v in (("Metformin", row.metformin), ("Sulfonylurea", row.sulfonylurea),
						   ("Insulin", row.insulin)) if v]
	st.write(f"**Medicines** {', '.join(meds) if meds else 'none'}")
	st.write(f"**Family history** {'yes' if row.family_history else 'no'} | "
			 f"**Genetic risk score** {row.polygenic_risk}")

# ---------------- forecast and alert ----------------
window = x[t - HISTORY:t][None].astype(np.float32)
now_val = g[t - 1]
with torch.no_grad():
	pred = model(torch.from_numpy(window), torch.from_numpy(s[None])).numpy()[0]
fc = now_val + pred * 100.0

with right:
	if now_val <= HIGH and fc.max() > HIGH:
		st.error(f"HIGH glucose alert: forecast peaks at {fc.max():.0f} mg/dL within 2 hours.")
	elif now_val >= LOW and fc.min() < LOW + MARGIN_LOW:
		st.error(f"LOW glucose alert: forecast drops to {fc.min():.0f} mg/dL within 2 hours.")
	elif now_val > HIGH:
		st.warning(f"Glucose already high ({now_val:.0f} mg/dL).")
	elif now_val < LOW:
		st.warning(f"Glucose already low ({now_val:.0f} mg/dL).")
	else:
		st.success(f"No alert. Forecast stays between {fc.min():.0f} and {fc.max():.0f} mg/dL.")

	hrs = np.arange(t - HISTORY, t + HORIZON) * 5 / 60
	fig, ax = plt.subplots(figsize=(9, 3.6))
	ax.plot(hrs[:HISTORY], g[t - HISTORY:t], color="gray", label="glucose (last 6 h)")
	ax.plot(hrs[HISTORY:], g[t:t + HORIZON], color="lightgray", ls=":", label="what actually happened")
	ax.plot(hrs[HISTORY:], fc, color="tab:red", label="2-hour forecast")
	ax.axhline(HIGH, color="orange", ls="--", lw=0.8)
	ax.axhline(LOW, color="blue", ls="--", lw=0.8)
	ax.set_xlabel("hours since start")
	ax.set_ylabel("glucose (mg/dL)")
	ax.legend(fontsize=8)
	fig.tight_layout()
	st.pyplot(fig)

# ---------------- what-if panel ----------------
st.divider()
st.subheader("What-if: remedy for a meal")
m = meals[(meals.patient_id == pid) & (meals.walked_after == 0)].copy()
m["t0"] = ((m.timestamp - START).dt.total_seconds() / 300).round().astype(int)
m = m[(m.t0 >= 1) & (m.t0 + AFTER >= HISTORY) & (m.t0 + AFTER + HORIZON <= n_steps)]
labels = [f"{(START + pd.Timedelta(minutes=5 * int(r.t0))).strftime('%d %b %H:%M')} - {r.meal} ({int(r.carbs_g)} g carbs)"
		  for r in m.itertuples()]
if not labels:
	st.info("No meals available for this patient.")
else:
	c1, c2 = st.columns(2)
	pick = c1.selectbox("Meal", labels, index=min(len(labels) - 1, 6))
	remedy = c2.selectbox("Remedy", REMEDIES)
	r = m.iloc[labels.index(pick)]
	t0, gi = int(r.t0), float(r.gi)
	base_w = x[t0 + AFTER - HISTORY:t0 + AFTER][None].astype(np.float32)
	var = make_variant(x, t0, gi, remedy)
	if var is None:
		st.info("This meal already has a low glycemic index, so this swap changes nothing.")
	else:
		f0 = forecast(model, base_w, s)[0]
		f1 = forecast(model, var[None].astype(np.float32), s)[0]
		k1, k2, k3 = st.columns(3)
		k1.metric("Predicted peak, no change", f"{f0.max():.0f} mg/dL")
		k2.metric("Predicted peak, with remedy", f"{f1.max():.0f} mg/dL", f"{f1.max() - f0.max():.0f} mg/dL")
		k3.metric("Minutes above 180 saved", f"{((f0 > HIGH).sum() - (f1 > HIGH).sum()) * 5}")
		hrs2 = np.arange(t0 + AFTER, t0 + AFTER + HORIZON) * 5 / 60
		fig2, ax2 = plt.subplots(figsize=(9, 3))
		ax2.plot(hrs2, f0, color="gray", label="as it happened (forecast)")
		ax2.plot(hrs2, f1, color="tab:green", label=f"with: {remedy}")
		ax2.axhline(HIGH, color="orange", ls="--", lw=0.8)
		ax2.set_xlabel("hours since start")
		ax2.set_ylabel("glucose (mg/dL)")
		ax2.legend(fontsize=8)
		fig2.tight_layout()
		st.pyplot(fig2)
		if "walk" in remedy:
			st.caption("Note: tested against the simulator, the model underestimates the effect of walking "
					   "(about 4 vs a true 13 mg/dL). Treat walk results as a conservative estimate.")
		st.caption("Portion-size and low-GI predictions were within about 0.2 mg/dL of the simulator's true effect on average.")
