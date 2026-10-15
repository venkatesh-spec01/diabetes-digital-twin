"""
Tests the 75% blend (picked on validation in walk_fix) on the 10 test patients.
Saves models/fusion_cf_blend75.pt, the model the dashboard uses.
"""
import pandas as pd
import torch
from train_baseline import DATA
from fusion_model import load_ehr, FusionForecaster
from whatif_finetune import truth_table, test_forecast_error

patients = pd.read_csv(f"{DATA}/patients.csv")
train_ids = patients[patients.group == "train"].patient_id.tolist()
test_ids = patients[patients.group == "test"].patient_id.tolist()
ehr = load_ehr(patients, train_ids)
pat = patients.set_index("patient_id")
meals = pd.read_csv(f"{DATA}/meals.csv", parse_dates=["timestamp"])

old = torch.load("models/fusion_lstm.pt")
tuned = torch.load("models/fusion_cf_v2.pt")
blend = {k: 0.25 * old[k] + 0.75 * tuned[k] for k in old}
torch.save(blend, "models/fusion_cf_blend75.pt")

m = FusionForecaster(10, ehr.shape[1])
m.load_state_dict(blend)
t = truth_table(m, test_ids, ehr, pat, meals)
e = test_forecast_error(m, test_ids, ehr)
print(f"TEST, 75% blend   (2-hour forecast RMSE {e.mean():.2f} mg/dL)")
print(t.to_string())
t["test_rmse_2h"] = round(e.mean(), 2)
t.to_csv("models/remedy_truth_step9b.csv")
print("Saved: models/fusion_cf_blend75.pt, models/remedy_truth_step9b.csv")
