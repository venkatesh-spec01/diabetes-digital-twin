"""
STEP 1 - Synthetic data generator for the Type 2 Diabetes digital twin.

Creates (all 100% synthetic, so no privacy / DPDP / HIPAA problem):
  data/patients.csv        -> one row per patient (the "static layer", like an EHR)
  data/meals.csv           -> every meal eaten (Indian meals)
  data/sensors/Pxxx.csv    -> glucose, steps, heart rate, HRV, sleep every 5 minutes
  data/summary.csv         -> quick health check of the generated data

Run:  python step1_generate_data.py
Needs: numpy, pandas  (runs on any normal laptop CPU in under a minute)
"""
import os
import numpy as np
import pandas as pd

# ----------------------------- SETTINGS ------------------------------------
SEED = 42              # same seed = same data every time (good for judges)
N_TRAIN_PATIENTS = 150 # random patients used to train the model later
DAYS = 14              # days of sensor data per patient
STEP_MIN = 5           # one reading every 5 minutes
START = pd.Timestamp("2026-09-01 00:00")
OUT = "data"

STEPS_PER_DAY = 1440 // STEP_MIN
N = DAYS * STEPS_PER_DAY

# ----------------------------- INDIAN MEAL MENU ----------------------------
# (name, carbs in grams, glycemic index)
MENU = {
    "breakfast": [("idli_sambar", 45, 70), ("poha", 50, 65), ("ragi_dosa", 35, 50),
                  ("aloo_paratha", 60, 70), ("oats_upma", 30, 55), ("sweet_pongal", 65, 78)],
    "lunch":     [("white_rice_dal_sabzi", 85, 75), ("roti_dal_sabzi", 60, 55),
                  ("millet_dal_sabzi", 55, 50), ("veg_biryani", 100, 70), ("curd_rice", 70, 68)],
    "snack":     [("chai_biscuit", 25, 65), ("samosa", 30, 60), ("fruit_bowl", 20, 50),
                  ("mithai_sweets", 45, 80), ("roasted_chana", 15, 35)],
    "dinner":    [("white_rice_dal_sabzi", 85, 75), ("roti_dal_sabzi", 60, 55),
                  ("millet_khichdi", 55, 50), ("veg_biryani", 100, 70), ("dosa_chutney", 50, 68)],
}
# how likely each diet type is to pick each dish (matched by dish name keywords)
DIET_BIAS = {
    "rice_heavy": ["rice", "biryani", "idli", "pongal", "poha"],
    "wheat":      ["roti", "paratha", "oats"],
    "millet":     ["millet", "ragi", "chana", "fruit", "oats"],
    "junk":       ["samosa", "mithai", "biryani", "paratha", "biscuit", "pongal"],
    "mixed":      [],
}
MEAL_OFFSET_H = {"breakfast": 1.5, "lunch": 7.0, "snack": 11.0, "dinner": 14.5}  # hours after waking


def pick_meal(rng, slot, diet):
    dishes = MENU[slot]
    w = np.array([3.0 if any(k in d[0] for k in DIET_BIAS[diet]) else 1.0 for d in dishes])
    return dishes[rng.choice(len(dishes), p=w / w.sum())]


# ----------------------------- PATIENT TABLE -------------------------------
def make_patient(rng, pid, group, **override):
    """Random India-style T2D patient. Any field can be overridden (used for hard test cases)."""
    age = int(rng.integers(25, 76))
    bmi = float(np.clip(rng.normal(26.5, 4.5), 18, 42))   # Indian T2D often at lower BMI
    years_dx = int(np.clip(rng.integers(0, 21), 0, age - 20))
    hba1c = float(np.clip(rng.normal(7.4 + 0.07 * years_dx, 1.1), 6.5, 11.5))
    fam = int(rng.random() < 0.6)
    prs = float(np.clip(0.3 + 0.35 * fam + rng.normal(0, 0.15), 0, 1))   # simple polygenic risk score
    on_ins = years_dx > 10 and hba1c > 8 and rng.random() < 0.4
    on_su = (not on_ins) and rng.random() < 0.3
    p = dict(
        patient_id=pid, group=group, age=age, sex=str(rng.choice(["M", "F"])),
        bmi=round(bmi, 1), years_since_diagnosis=years_dx, hba1c=round(hba1c, 1),
        ldl=int(rng.normal(110, 30)), triglycerides=int(np.clip(rng.normal(170, 60), 70, 450)),
        family_history=fam, polygenic_risk=round(prs, 2),
        metformin=int(rng.random() < 0.9), sulfonylurea=int(on_su), insulin=int(on_ins),
        diet=str(rng.choice(list(DIET_BIAS))), activity=str(rng.choice(["sedentary", "light", "active"], p=[.4, .4, .2])),
        sleep_mean_h=round(float(np.clip(rng.normal(6.8, 0.8), 4.5, 8.5)), 1),
        wake_hour=6.5, resting_hr=int(rng.integers(60, 82)),
        walk_after_meal_prob=round(float(rng.uniform(0.05, 0.5)), 2),
        meal_skip_prob=round(float(rng.uniform(0.0, 0.1)), 2), meal_size=1.0,
        dose_miss_prob=0.05, note="random patient",
    )
    p.update(override)
    # hidden "insulin resistance" factor: how hard the body reacts to carbs
    ir = 0.7 + 0.04 * (p["bmi"] - 22) + 0.08 * (p["hba1c"] - 6.5) + 0.3 * p["polygenic_risk"] + rng.normal(0, 0.1)
    p.setdefault("insulin_resistance", None)
    if p["insulin_resistance"] is None:
        p["insulin_resistance"] = round(float(np.clip(ir, 0.5, 2.2)), 2)
    p["fasting_glucose"] = p.get("fasting_glucose") or int(0.8 * (28.7 * p["hba1c"] - 46.7) - 12)
    # how strongly the medicine can push glucose DOWN too far (hypoglycemia risk)
    p["hypo_strength"] = round(0.35 * p["sulfonylurea"] + 0.75 * p["insulin"], 2)
    return p


def hard_test_patients(rng):
    """10 deliberately different / opposite / difficult patients for final testing."""
    T = lambda i, **kw: make_patient(rng, f"T{i:02d}", "test", **kw)
    return [
        T(1, age=27, bmi=21.5, years_since_diagnosis=0, hba1c=6.6, metformin=1, sulfonylurea=0, insulin=0,
          diet="millet", activity="active", sleep_mean_h=7.8, walk_after_meal_prob=0.8, meal_skip_prob=0.0,
          note="young, newly diagnosed, athletic, well controlled"),
        T(2, age=74, bmi=24.0, years_since_diagnosis=20, hba1c=10.4, metformin=0, sulfonylurea=0, insulin=1,
          diet="rice_heavy", activity="sedentary", sleep_mean_h=5.5, walk_after_meal_prob=0.0, meal_skip_prob=0.12,
          dose_miss_prob=0.2, note="elderly, 20 years diabetic, insulin, poor control, skips meals"),
        T(3, age=45, bmi=38.5, years_since_diagnosis=6, hba1c=9.2, metformin=1, sulfonylurea=0, insulin=0,
          polygenic_risk=0.9, family_history=1, diet="rice_heavy", activity="sedentary", sleep_mean_h=6.0,
          walk_after_meal_prob=0.02, meal_size=1.3, note="severely obese, strong insulin resistance, rice heavy"),
        T(4, age=52, bmi=20.5, years_since_diagnosis=8, hba1c=7.2, metformin=1, sulfonylurea=1, insulin=0,
          diet="wheat", activity="light", sleep_mean_h=7.0, meal_skip_prob=0.2,
          note="lean (thin-fat) Indian patient on sulfonylurea, skips meals, hypo risk"),
        T(5, age=38, bmi=29.0, years_since_diagnosis=3, hba1c=8.1, metformin=1, sulfonylurea=0, insulin=0,
          wake_hour=15.0, sleep_mean_h=4.5, diet="junk", activity="light", note="night-shift worker, 4.5h sleep, junk food"),
        T(6, age=60, bmi=25.5, years_since_diagnosis=12, hba1c=7.0, metformin=1, sulfonylurea=0, insulin=1,
          diet="millet", activity="active", walk_after_meal_prob=0.9, meal_skip_prob=0.15, sleep_mean_h=7.5,
          note="super healthy lifestyle BUT on insulin, so walking + skipped meals cause lows"),
        T(7, age=41, bmi=24.5, years_since_diagnosis=2, hba1c=6.8, metformin=1, sulfonylurea=0, insulin=0,
          polygenic_risk=0.95, family_history=1, diet="mixed", activity="light", sleep_mean_h=7.0,
          note="mild now, but very high genetic risk"),
        T(8, age=29, bmi=31.5, years_since_diagnosis=1, hba1c=8.8, metformin=0, sulfonylurea=0, insulin=0,
          diet="junk", activity="sedentary", sleep_mean_h=5.8, walk_after_meal_prob=0.0, meal_size=1.2,
          note="early-onset, no medicine, junk food, no exercise"),
        T(9, age=58, bmi=28.0, years_since_diagnosis=9, hba1c=7.9, ldl=165, triglycerides=320, metformin=1,
          sulfonylurea=1, insulin=0, diet="mixed", activity="light", meal_skip_prob=0.35, sleep_mean_h=6.3,
          note="high cholesterol, skips breakfast often, irregular eating"),
        T(10, age=50, bmi=33.0, years_since_diagnosis=14, hba1c=9.8, metformin=1, sulfonylurea=0, insulin=1,
          diet="junk", activity="sedentary", meal_size=1.5, dose_miss_prob=0.4, meal_skip_prob=0.1,
          sleep_mean_h=5.0, walk_after_meal_prob=0.0,
          note="festival-week eating, huge sweet meals, misses insulin doses: wild swings"),
    ]


# ----------------------------- SENSOR SIMULATION ---------------------------
def kernel(t_min, peak):
    """Smooth 'rise then fall' curve, 0 before meal, max (=1) at `peak` minutes."""
    t = np.maximum(t_min, 0)
    return (t / peak) * np.exp(1 - t / peak)


def simulate(p, rng):
    idx = np.arange(N)
    minutes = idx * STEP_MIN
    hour = (minutes / 60) % 24

    # ---- sleep: each day wake time + sleep length ----
    sleeping = np.zeros(N, dtype=int)
    sleep_len = np.zeros(DAYS)
    for d in range(DAYS):
        dur = float(np.clip(rng.normal(p["sleep_mean_h"], 0.7), 3.0, 9.5))
        sleep_len[d] = dur
        wake = d * 24 + p["wake_hour"] + rng.normal(0, 0.3)
        a, b = int((wake - dur) * 60 / STEP_MIN), int(wake * 60 / STEP_MIN)
        sleeping[max(a, 0):min(max(b, 0), N)] = 1
    day_of = (minutes // 1440).astype(int)
    prev_sleep = np.where(day_of > 0, sleep_len[np.maximum(day_of - 1, 0)], sleep_len[0])
    deficit = np.clip(7.0 - prev_sleep, 0, 4)       # hours of missing sleep

    # ---- steps ----
    base_rate = {"sedentary": 8, "light": 22, "active": 45}[p["activity"]]
    steps = rng.poisson(base_rate * rng.lognormal(0, 0.4, N)).astype(float)
    for d in range(DAYS):                            # a few random walking bouts per day
        for _ in range(int(rng.integers(1, 4 if p["activity"] != "sedentary" else 2))):
            s = int(d * STEPS_PER_DAY + rng.integers(100, 250))
            steps[s:s + int(rng.integers(2, 5))] += rng.normal(420, 60)
    steps[sleeping == 1] = rng.poisson(0.3, int(sleeping.sum()))

    # ---- meals + glucose ----
    ir = p["insulin_resistance"] * (0.85 if p["metformin"] else 1.0) * (1 + 0.05 * deficit)
    baseline = p["fasting_glucose"] + 8 * np.exp(-((hour - 6) / 2) ** 2) + 6 * deficit   # dawn effect + bad sleep
    baseline = baseline - 3 * p["metformin"] - 14 * p["hypo_strength"]   # medicines also pull the baseline down
    glucose = baseline.copy()
    meals = []
    for d in range(DAYS):
        wake = d * 24 + p["wake_hour"]
        for slot, off in MEAL_OFFSET_H.items():
            if rng.random() < p["meal_skip_prob"]:
                # medicine is still taken even if the meal is skipped -> classic cause of a low
                t_skip = int((wake + off) * 60 / STEP_MIN)
                if p["hypo_strength"] > 0 and 0 <= t_skip < N and rng.random() < 0.6:
                    seg = np.arange(min(72, N - t_skip)) * STEP_MIN
                    glucose[t_skip:t_skip + len(seg)] -= p["hypo_strength"] * 45 * kernel(seg, 120)
                continue
            name, carbs, gi = pick_meal(rng, slot, p["diet"])
            carbs = carbs * p["meal_size"] * rng.uniform(0.85, 1.15)
            t0 = int((wake + off + rng.normal(0, 0.4)) * 60 / STEP_MIN)
            if t0 < 0 or t0 >= N:
                continue
            walked = int(rng.random() < p["walk_after_meal_prob"])
            if walked:                                   # 10-min walk starting 15 min after the meal
                steps[t0 + 3:t0 + 5] += rng.normal(450, 50)
            seg = np.arange(min(72, N - t0)) * STEP_MIN   # next 6 hours
            amp = 0.6 * carbs * (gi / 70) * ir[t0] * (0.72 if walked else 1.0)
            glucose[t0:t0 + len(seg)] += amp * kernel(seg, 45)
            missed = rng.random() < p["dose_miss_prob"]
            if p["hypo_strength"] > 0 and not missed:    # medicine pushes glucose down ~2.5h later
                glucose[t0:t0 + len(seg)] -= p["hypo_strength"] * 0.85 * carbs * (gi / 70) * kernel(seg, 150)
            meals.append(dict(patient_id=p["patient_id"], timestamp=START + pd.Timedelta(minutes=int(t0 * STEP_MIN)),
                              slot=slot, meal=name, carbs_g=round(carbs), gi=gi, walked_after=walked))

    # sensor noise: slow wobble + small jitter
    wobble = np.zeros(N)
    for i in range(1, N):
        wobble[i] = 0.9 * wobble[i - 1] + rng.normal(0, 2.0)
    glucose = np.clip(glucose + wobble, 40, 400)

    # ---- heart rate and HRV ----
    hr = p["resting_hr"] + 0.06 * steps - 8 * sleeping + rng.normal(0, 2, N)
    for m in meals:                                      # small bump after eating
        t0 = int((m["timestamp"] - START).total_seconds() // 60 // STEP_MIN)
        hr[t0:t0 + 6] += 3
    hrv_base = float(np.clip(55 - 0.4 * (p["age"] - 30) - 0.5 * (p["bmi"] - 22), 15, 60))
    hrv = hrv_base * np.where(sleeping == 1, 1.3, 0.8) * (1 - 0.04 * deficit) + rng.normal(0, 2, N)

    df = pd.DataFrame(dict(
        timestamp=START + pd.to_timedelta(minutes, unit="m"),
        glucose_mgdl=glucose.round(1), steps=steps.clip(0).round().astype(int),
        heart_rate=hr.clip(40, 180).round().astype(int), hrv_ms=hrv.clip(5, 120).round(1),
        sleeping=sleeping))
    meal_df = pd.DataFrame(meals)
    # put meal carbs on the sensor timeline too (handy for the model in step 3)
    df["carbs_g"] = 0
    df["meal_gi"] = 0
    for m in meals:
        i = int((m["timestamp"] - START).total_seconds() // 60 // STEP_MIN)
        df.loc[i, ["carbs_g", "meal_gi"]] = [m["carbs_g"], m["gi"]]
    return df, meal_df


# ----------------------------- MAIN ----------------------------------------
def main():
    os.makedirs(f"{OUT}/sensors", exist_ok=True)
    rng = np.random.default_rng(SEED)
    N_ORIGINAL = 40   # the first 40 patients and T01-T10 stay identical to the first run
    patients = [make_patient(rng, f"P{i:03d}", "train") for i in range(1, N_ORIGINAL + 1)]
    patients += hard_test_patients(rng)
    extra_rng = np.random.default_rng(SEED + 1)   # separate random stream for the new patients
    patients += [make_patient(extra_rng, f"P{i:03d}", "train") for i in range(N_ORIGINAL + 1, N_TRAIN_PATIENTS + 1)]
    
    pd.DataFrame(patients).to_csv(f"{OUT}/patients.csv", index=False)

    all_meals, rows = [], []
    for p in patients:
        sim_rng = np.random.default_rng(SEED + sum(map(ord, p["patient_id"])))
        df, meal_df = simulate(p, sim_rng)
        df.to_csv(f"{OUT}/sensors/{p['patient_id']}.csv", index=False)
        all_meals.append(meal_df)
        g = df["glucose_mgdl"]
        rows.append(dict(patient_id=p["patient_id"], group=p["group"],
                         mean_glucose=round(g.mean()), pct_in_range_70_180=round(100 * ((g >= 70) & (g <= 180)).mean(), 1),
                         pct_high_over_180=round(100 * (g > 180).mean(), 1), pct_low_under_70=round(100 * (g < 70).mean(), 1),
                         min_glucose=round(g.min()), max_glucose=round(g.max()), mean_steps_per_day=round(df.steps.sum() / DAYS)))
    pd.concat(all_meals).to_csv(f"{OUT}/meals.csv", index=False)
    summary = pd.DataFrame(rows)
    summary.to_csv(f"{OUT}/summary.csv", index=False)
    print(f"Done. {len(patients)} patients, {DAYS} days each, saved in '{OUT}/'.\n")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
