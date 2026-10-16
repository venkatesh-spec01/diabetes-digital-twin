# makes the sample medicine table (data/medications.csv) for the dashboard.
# each patient gets medicines based on their record (metformin, glimepiride,
# insulin, statin) with a typical dose. all of it is synthetic, not a prescription.
# run it after make_data.py: python make_medications.py
import pandas as pd

DATA = "data"
# read the patients made earlier
p = pd.read_csv(f"{DATA}/patients.csv")


# converts miss probability into a text label
def status(prob):
    if prob >= 0.30:
        return "Often missed"
    if prob >= 0.12:
        return "Sometimes missed"
    return "Taking regularly"


rows = []
# make medicine rows from each patient's record
for r in p.itertuples():
    st = status(r.dose_miss_prob)
    # metformin dose depends on hba1c
    if r.metformin:
        rows.append((r.patient_id, "Metformin", "1000 mg" if r.hba1c >= 8.5 else "500 mg", "Twice daily",
                     "With or just after breakfast and dinner", st, ""))
    if r.sulfonylurea:
        rows.append((r.patient_id, "Glimepiride (sulfonylurea)", "2 mg" if r.hba1c >= 8.0 else "1 mg", "Once daily",
                     "With the first main meal. Do not take if skipping the meal", st,
                     "Can cause low glucose"))
    if r.insulin:
        # insulin units depend on hba1c, 10 to 24
        units = int(min(24, max(10, round(10 + (r.hba1c - 7.0) * 2))))
        rows.append((r.patient_id, "Insulin glargine (basal)", f"{units} units", "Once daily",
                     "Same time every day, at bedtime", st, "Can cause low glucose"))
    if r.ldl >= 100:
        rows.append((r.patient_id, "Atorvastatin", "10 mg" if r.ldl < 130 else "20 mg", "Once daily", "At night", st, ""))

df = pd.DataFrame(rows, columns=["patient_id", "drug", "dose", "frequency", "when_to_take", "taking_status", "hospital_note"])
# add a hospital column to the table
df.insert(5, "prescribed_by", "Hospital OPD (synthetic sample)")
df = df[["patient_id", "drug", "dose", "frequency", "when_to_take", "prescribed_by", "taking_status", "hospital_note"]]
# saves medicines for the dashboard
df.to_csv(f"{DATA}/medications.csv", index=False)
print(f"Saved {DATA}/medications.csv with {len(df)} rows for {df.patient_id.nunique()} patients")
