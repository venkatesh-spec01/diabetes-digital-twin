# logic for the doctor dashboard panels (no streamlit in this file).
#   zone_of: which glucose zone a value is in
#   residual sigma: how wrong the forecast usually is for this patient
#   safety status: alert status with an uncertainty band
#   load meds, save meds: the medicine table
#   load diet, save diet: the diet chart
#   summary stats, make pdf: the one page patient summary
# all patients and medicines are synthetic samples.
import io
import json
import os
import textwrap

import numpy as np
import pandas as pd

from clinical_reference import ZONES, DIET_PLANS, DIET_COLUMNS

# glucose limits used for alerts
HIGH, LOW = 180.0, 70.0
# multiplier that covers about 80 percent of errors
Z80 = 1.28
# above this error we call it low confidence
LOW_CONFIDENCE_SIGMA = 25

# columns of the medicine table
MED_COLUMNS = ["Medicine", "Dose", "How often", "When to take", "Prescribed by", "Patient reports", "Note"]


# tells which glucose zone a value is in
def zone_of(v):
    for lo, hi, name, colour in ZONES:
        if lo <= v < hi:
            return name, colour
    return ZONES[-1][2], ZONES[-1][3]


# usual forecast error for this patient
def residual_sigma(g, ms, F, horizon=24, floor=3.0):
    errs = []
    for m, f in zip(ms, F):
        m = int(m)
        if m + horizon <= len(g):
            errs.append(np.asarray(g[m:m + horizon], dtype=float) - np.asarray(f, dtype=float))
    # not enough history, use a typical error
    if len(errs) < 5:
        return np.linspace(8.0, 20.0, horizon)
    return np.maximum(np.std(np.stack(errs), axis=0), floor)


# range around the forecast line
def band(fc, sigma, z=Z80):
    fc = np.asarray(fc, dtype=float)
    return fc - z * sigma, fc + z * sigma


# decides alert status using the uncertainty band
def safety_status(now, fc, sigma, low_margin=0.0, z=Z80):
    fc = np.asarray(fc, dtype=float)
    lo_b, hi_b = band(fc, sigma, z)
    s2h = float(sigma[-1])
    conf = "Lower confidence" if s2h > LOW_CONFIDENCE_SIGMA else "Normal confidence"
    out = dict(kind="OK", possible=False, label="Clear", confidence=conf, sigma_2h=s2h,
               reason="The forecast and its uncertainty band stay inside 70-180 mg/dL.")
    # check current glucose first, then the forecast
    if now > HIGH:
        out.update(kind="HIGH", label="Already high",
                   reason=f"Glucose is already {now:.0f} mg/dL, above 180.")
    elif now < LOW:
        out.update(kind="LOW", label="Already low",
                   reason=f"Glucose is already {now:.0f} mg/dL, below 70.")
    elif fc.max() > HIGH:
        out.update(kind="HIGH", label="High glucose expected",
                   reason=f"The forecast peaks at {fc.max():.0f} mg/dL, above 180.")
    elif fc.min() < LOW + low_margin:
        out.update(kind="LOW", label="Low glucose expected",
                   reason=f"The forecast drops to {fc.min():.0f} mg/dL, close to or below 70.")
    # only the band crosses, so it is borderline
    elif hi_b.max() > HIGH:
        out.update(kind="HIGH", possible=True, label="Possible high (borderline)",
                   reason=f"The forecast peaks at {fc.max():.0f} mg/dL, but with the usual error for this patient "
                          f"(+/- {z * s2h:.0f} mg/dL at 2 hours) it could reach {hi_b.max():.0f}.")
    elif lo_b.min() < LOW:
        out.update(kind="LOW", possible=True, label="Possible low (borderline)",
                   reason=f"The forecast bottoms at {fc.min():.0f} mg/dL, but with the usual error for this patient "
                          f"(+/- {z * s2h:.0f} mg/dL at 2 hours) it could fall to {lo_b.min():.0f}.")
    return out


# loads medicine list, saved edits first
def load_meds(pid, data_dir, notes_dir="notes"):
    saved = os.path.join(notes_dir, f"meds_{pid}.json")
    if os.path.exists(saved):
        try:
            with open(saved, "r", encoding="utf-8") as f:
                df = pd.DataFrame(json.load(f))
            for c in MED_COLUMNS:
                if c not in df.columns:
                    df[c] = ""
            return df[MED_COLUMNS], True
        except Exception:
            pass
    path = os.path.join(data_dir, "medications.csv")
    if os.path.exists(path):
        m = pd.read_csv(path)
        m = m[m.patient_id == pid]
        df = pd.DataFrame({
            "Medicine": m.drug, "Dose": m.dose, "How often": m.frequency, "When to take": m.when_to_take,
            "Prescribed by": m.prescribed_by, "Patient reports": m.taking_status, "Note": m.hospital_note.fillna("")})
        return df.reset_index(drop=True)[MED_COLUMNS], False
    return pd.DataFrame(columns=MED_COLUMNS), False


# saves the doctor's medicine edits to a file
def save_meds(pid, df, notes_dir="notes"):
    os.makedirs(notes_dir, exist_ok=True)
    clean = df.fillna("").astype(str)
    clean = clean[clean["Medicine"].str.strip() != ""]
    with open(os.path.join(notes_dir, f"meds_{pid}.json"), "w", encoding="utf-8") as f:
        json.dump(clean.to_dict(orient="records"), f, indent=2)


# lists medicines the patient is missing
def adherence_note(df):
    if len(df) == 0 or "Patient reports" not in df.columns:
        return None
    # keep rows not taken regularly
    bad = df[~df["Patient reports"].astype(str).str.lower().str.contains("regular")]
    bad = bad[bad["Patient reports"].astype(str).str.strip() != ""]
    if len(bad) == 0:
        return None
    return "; ".join(f"{row['Medicine']}: {row['Patient reports']}" for _, row in bad.iterrows())


# turns a diet plan into a table
def diet_table(plan):
    return pd.DataFrame(DIET_PLANS[plan], columns=DIET_COLUMNS)


# loads the saved diet chart if there is one
def load_diet(pid, plan, notes_dir="notes"):
    saved = os.path.join(notes_dir, f"diet_{pid}.json")
    if os.path.exists(saved):
        try:
            with open(saved, "r", encoding="utf-8") as f:
                d = json.load(f)
            return pd.DataFrame(d["rows"], columns=DIET_COLUMNS), d.get("plan", plan), True
        except Exception:
            pass
    return diet_table(plan), plan, False


# saves the edited diet chart
def save_diet(pid, plan, df, notes_dir="notes"):
    os.makedirs(notes_dir, exist_ok=True)
    with open(os.path.join(notes_dir, f"diet_{pid}.json"), "w", encoding="utf-8") as f:
        json.dump({"plan": plan, "rows": df.fillna("").astype(object).values.tolist()}, f, indent=2)


# numbers for the patient summary page
def summary_stats(gl, ms, F, g, low_margin=0.0):
    gl = np.asarray(gl, dtype=float)
    n_high = n_low = 0
    for m, f in zip(ms, F):
        now = g[int(m) - 1]
        if now <= HIGH and np.max(f) > HIGH:
            n_high += 1
        if now >= LOW and np.min(f) < LOW + low_margin:
            n_low += 1
    return dict(tir=float(((gl >= LOW) & (gl <= HIGH)).mean() * 100), above=float((gl > HIGH).mean() * 100),
                below=float((gl < LOW).mean() * 100), mean=float(gl.mean()), cv=float(gl.std() / gl.mean() * 100),
                n_high=n_high, n_low=n_low, n_moments=len(ms))


# builds the one page summary as a pdf
def make_pdf(info):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    st = info["stats"]
    # page is A4 size, built with matplotlib
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.text(0.07, 0.955, "Type 2 Diabetes Digital Twin: patient summary", fontsize=15, fontweight="bold")
    fig.text(0.07, 0.932, f"Synthetic demonstration data. Prepared {info['today']}. Not medical advice.", fontsize=8.5, color="#555")
    y = 0.895
    head = (f"Patient {info['pid']}   |   {info['age']} years, {'male' if info['sex'] == 'M' else 'female'}   |   "
            f"BMI {info['bmi']}   |   HbA1c {info['hba1c']}%   |   fasting glucose {info['fasting']} mg/dL   |   "
            f"{info['years']} years since diagnosis")
    for line in textwrap.wrap(head, 105):
        fig.text(0.07, y, line, fontsize=9.5, fontweight="bold")
        y -= 0.02
    y -= 0.01

    # writes a heading and its lines
    def block(title, lines, y):
        fig.text(0.07, y, title, fontsize=11, fontweight="bold", color="#1f3a5f")
        y -= 0.022
        for ln in lines:
            for w in textwrap.wrap(ln, 110) or [""]:
                fig.text(0.08, y, w, fontsize=9)
                y -= 0.0175
        return y - 0.012

    y = block("Glucose over the whole record", [
        f"Time in range 70-180 mg/dL: {st['tir']:.0f}%   |   above 180: {st['above']:.0f}%   |   below 70: {st['below']:.1f}%",
        f"Average glucose {st['mean']:.0f} mg/dL, variability (CV) {st['cv']:.0f}%",
        f"Forecast alerts raised: {st['n_high']} high, {st['n_low']} low, out of {st['n_moments']} moments checked",
        f"Typical forecast error at 2 hours for this patient: about {info['sigma2h']:.0f} mg/dL"], y)
    y = block("Flags from the record", ["; ".join(info["flags"]) if info["flags"] else "none"], y)

    med_lines = []
    for r in info["meds"].itertuples(index=False):
        med_lines.append(f"- {r[0]}: {r[1]}, {r[2]}. {r[3]}. Patient reports: {r[5]}." + (f" Note: {r[6]}" if str(r[6]).strip() else ""))
    y = block("Medicines (hospital prescription and what the patient reports)", med_lines or ["none recorded"], y)
    y = block("Diet chart given", [f"Sample plan: {info['diet_plan']}. See the dashboard for the full chart."], y)

    ax = fig.add_axes([0.1, 0.1, 0.82, 0.22])
    ax.plot(info["ts"], info["gl"], color="#444", lw=0.6)
    ax.axhspan(LOW, HIGH, color="#2ca02c", alpha=0.12)
    ax.axhline(HIGH, color="#f59e0b", lw=0.8, ls="--")
    ax.axhline(LOW, color="#3b82f6", lw=0.8, ls="--")
    ax.set_ylabel("glucose (mg/dL)")
    ax.set_title("Whole record (green band = 70-180)", fontsize=9)
    ax.tick_params(labelsize=7)
    fig.autofmt_xdate()
    fig.text(0.07, 0.04, "Decision support prototype built on synthetic data. It does not replace the treating doctor.",
             fontsize=8, color="#777")
    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        pdf.savefig(fig)
    plt.close(fig)
    return buf.getvalue()
