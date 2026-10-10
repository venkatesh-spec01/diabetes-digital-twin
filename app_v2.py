"""
Doctor dashboard v2 for the Type 2 Diabetes digital twin (100% synthetic data).

Run:  streamlit run app_v2.py
"""
import numpy as np
import pandas as pd
import torch
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from step2_train_model import DATA, HISTORY, HORIZON, load_patient
from step3_fusion import load_ehr, FusionForecaster
from step5_remedies import make_variant, forecast, AFTER, REMEDIES, START

HIGH, LOW = 180.0, 70.0
GREEN, RED, BLUE, ORANGE, GRAY = "#2ca02c", "#e4572e", "#3b82f6", "#f59e0b", "#9ca3af"

st.set_page_config(page_title="T2D Digital Twin", page_icon="🩺", layout="wide")

st.markdown('''
<style>
html, [data-testid="stMain"], section.main { scroll-behavior: smooth; }
@keyframes fadeUp { from { opacity: 0; transform: translateY(14px); } to { opacity: 1; transform: none; } }
.block-container { padding-top: 2rem; animation: fadeUp .5s ease-out; }
div[data-testid="stMetric"] {
    background: rgba(120,130,150,.12); border-radius: 12px; padding: 12px 16px;
    transition: transform .2s ease, box-shadow .2s ease;
}
div[data-testid="stMetric"]:hover { transform: translateY(-3px); box-shadow: 0 6px 18px rgba(0,0,0,.25); }
.stPlotlyChart { animation: fadeUp .7s ease-out; }
.badge { display: inline-block; padding: 3px 10px; margin: 2px 6px 2px 0; border-radius: 999px; font-size: 0.82rem; }
</style>
''', unsafe_allow_html=True)


# ----------------------------- loading -----------------------------
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


@st.cache_data
def load_raw(pid):
    return pd.read_csv(f"{DATA}/sensors/{pid}.csv", parse_dates=["timestamp"])


@st.cache_data
def get_patient(pid):
    return load_patient(pid)


@st.cache_data
def day_forecasts(pid, start, stop, step):
    x, g = get_patient(pid)
    s = ehr.loc[pid].values.astype(np.float32)
    ms = list(range(start, stop, step))
    W = np.stack([x[m - HISTORY:m] for m in ms]).astype(np.float32)
    S = np.repeat(s[None], len(ms), axis=0)
    with torch.no_grad():
        pred = model(torch.from_numpy(W), torch.from_numpy(S)).numpy()
    now = g[np.array(ms) - 1]
    return ms, now[:, None] + pred * 100.0


# ----------------------------- sidebar -----------------------------
with st.sidebar:
    st.header("Controls")
    pid = st.selectbox("Patient (unseen during training)", test_ids)
    day = st.slider("Day to review", 1, 14, 4)
    low_sens = st.radio("Low-glucose alert sensitivity", ["High (recommended)", "Standard"], index=0)
    st.caption("High sensitivity catches more lows but gives more false alarms. "
               "A missed low is the more dangerous mistake.")

LOW_MARGIN = 15.0 if low_sens.startswith("High") else 0.0


def alert_of(now, fc):
    if now <= HIGH and fc.max() > HIGH:
        return "HIGH"
    if now >= LOW and fc.min() < LOW + LOW_MARGIN:
        return "LOW"
    return "OK"


row = patients[patients.patient_id == pid].iloc[0]
raw = load_raw(pid)
x, g = get_patient(pid)
s = ehr.loc[pid].values.astype(np.float32)
n_steps = len(g)
ts = pd.DatetimeIndex(raw["timestamp"])


def badge(text, color):
    return f'<span class="badge" style="background:{color}22;border:1px solid {color};color:{color}">{text}</span>'


def flag_list(r):
    out = []
    if r.hba1c >= 7.0:
        out.append(("HbA1c at or above 7%", ORANGE))
    if r.ldl >= 100:
        out.append(("LDL at or above 100", ORANGE))
    if r.triglycerides >= 150:
        out.append(("Triglycerides at or above 150", ORANGE))
    if r.bmi >= 23:
        out.append(("BMI at or above 23 (Asian overweight cut-off)", ORANGE))
    if r.family_history:
        out.append(("Family history of diabetes", BLUE))
    if r.insulin or r.sulfonylurea:
        out.append(("Medicine can cause low glucose", RED))
    return out


# ----------------------------- header -----------------------------
st.markdown("## 🩺 Type 2 Diabetes Digital Twin")
st.caption("Prototype on 100% synthetic data. Not medical advice. Needs clinical validation.")

cols = st.columns(6)
cols[0].metric("Patient", pid, f"{row.age} y, {row.sex}", delta_color="off")
cols[1].metric("BMI", f"{row.bmi}")
cols[2].metric("HbA1c", f"{row.hba1c}%")
cols[3].metric("Fasting glucose", f"{row.fasting_glucose} mg/dL")
cols[4].metric("Years since diagnosis", f"{row.years_since_diagnosis}")
cols[5].metric("Genetic risk score", f"{row.polygenic_risk}")

meds = [n for n, v in (("Metformin", row.metformin), ("Sulfonylurea", row.sulfonylurea),
                       ("Insulin", row.insulin)) if v]
html = "".join(badge(m, GREEN) for m in meds) or badge("No diabetes medicine", GRAY)
html += "".join(badge(t, c) for t, c in flag_list(row))
st.markdown(html, unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["Live twin", "14-day patterns", "What-if remedies", "Patient record"])

# ============================ TAB 1: live twin ============================
with tab1:
    start = max((day - 1) * 288, HISTORY)
    stop = min(day * 288, n_steps - HORIZON - 1)
    ms, F = day_forecasts(pid, start, stop, 12)
    alerts = [alert_of(g[m - 1], F[i]) for i, m in enumerate(ms)]
    sym = {"HIGH": " ▲", "LOW": " ▼", "OK": ""}
    labels = [ts[m - 1].strftime("%H:%M") + sym[a] for m, a in zip(ms, alerts)]

    k = st.columns(4)
    k[0].metric("Moments reviewed", len(ms))
    k[1].metric("High-glucose alerts ▲", alerts.count("HIGH"))
    k[2].metric("Low-glucose alerts ▼", alerts.count("LOW"))
    k[3].metric("Highest forecast", f"{F.max():.0f} mg/dL")

    st.markdown(f"##### Day {day}: press Play, or drag the slider to move the twin through the day")

    def traces_for(m, fc):
        a, b = m - HISTORY, m + HORIZON
        return [
            (ts[a:m], g[a:m]),
            (ts[m - 1:b], g[m - 1:b]),
            (ts[m - 1:b], np.r_[g[m - 1], fc]),
            (ts[m - 1:m], g[m - 1:m]),
        ]

    first = traces_for(ms[0], F[0])
    base = [
        go.Scatter(x=first[0][0], y=first[0][1], name="Glucose (last 6 h)",
                   line=dict(color=GRAY, width=3),
                   hovertemplate="%{x|%H:%M}  %{y:.0f} mg/dL<extra></extra>"),
        go.Scatter(x=first[1][0], y=first[1][1], name="What actually happened (hidden from model)",
                   line=dict(color=GRAY, width=2, dash="dot"),
                   hovertemplate="%{x|%H:%M}  %{y:.0f} mg/dL<extra></extra>"),
        go.Scatter(x=first[2][0], y=first[2][1], name="2-hour forecast",
                   line=dict(color=RED, width=3),
                   hovertemplate="%{x|%H:%M}  %{y:.0f} mg/dL<extra></extra>"),
        go.Scatter(x=first[3][0], y=first[3][1], name="Now", mode="markers",
                   marker=dict(color=RED, size=11, line=dict(color="white", width=2))),
    ]
    frames = []
    for i, m in enumerate(ms):
        a, b = m - HISTORY, m + HORIZON
        data = [go.Scatter(x=xx, y=yy) for xx, yy in traces_for(m, F[i])]
        frames.append(go.Frame(name=str(i), traces=[0, 1, 2, 3], data=data,
                               layout=go.Layout(xaxis=dict(range=[str(ts[a]), str(ts[b - 1])]))))

    lo = min(float(g[ms[0] - HISTORY:ms[-1] + HORIZON].min()), float(F.min()), 60.0) - 10
    hi = max(float(g[ms[0] - HISTORY:ms[-1] + HORIZON].max()), float(F.max()), 200.0) + 15

    fig = go.Figure(data=base, frames=frames)
    fig.add_hrect(y0=LOW, y1=HIGH, fillcolor=GREEN, opacity=0.07, line_width=0)
    fig.add_hline(y=HIGH, line_dash="dash", line_color=ORANGE, line_width=1)
    fig.add_hline(y=LOW, line_dash="dash", line_color=BLUE, line_width=1)
    play_args = dict(frame=dict(duration=700, redraw=False),
                     transition=dict(duration=500, easing="cubic-in-out"),
                     fromcurrent=True, mode="immediate")
    fig.update_layout(
        height=560, margin=dict(l=10, r=10, t=40, b=130), hovermode="x unified",
        legend=dict(orientation="h", y=1.12),
        yaxis=dict(title="glucose (mg/dL)", range=[lo, hi]),
        xaxis=dict(range=[str(ts[ms[0] - HISTORY]), str(ts[ms[0] + HORIZON - 1])],
                   tickformat="%H:%M<br>%d %b"),
        updatemenus=[dict(type="buttons", direction="left", x=0.0, y=-0.12, xanchor="left", yanchor="top",
                          showactive=False, pad=dict(t=40),
                          buttons=[dict(label="▶ Play", method="animate", args=[None, play_args]),
                                   dict(label="⏸ Pause", method="animate",
                                        args=[[None], dict(frame=dict(duration=0, redraw=False),
                                                           transition=dict(duration=0), mode="immediate")])])],
        sliders=[dict(active=0, x=0.2, len=0.8, y=-0.12, yanchor="top", pad=dict(t=40),
                      currentvalue=dict(prefix="Time: ", font=dict(size=14)),
                      steps=[dict(method="animate", label=labels[i],
                                  args=[[str(i)], dict(mode="immediate", frame=dict(duration=0, redraw=False),
                                                       transition=dict(duration=400, easing="cubic-in-out"))])
                             for i in range(len(ms))])],
    )
    st.plotly_chart(fig, key="live_chart")
    st.caption("▲ = high-glucose alert at that moment, ▼ = low-glucose alert. The dotted line is the real "
               "glucose that followed, which the model never sees.")

    rows = [{"Time": ts[m - 1].strftime("%d %b %H:%M"), "Glucose now": round(float(g[m - 1])),
             "Forecast peak": round(float(F[i].max())), "Forecast low": round(float(F[i].min())),
             "Alert": alerts[i]}
            for i, m in enumerate(ms) if alerts[i] != "OK"]
    if rows:
        st.markdown("##### Alerts raised on this day")
        st.dataframe(pd.DataFrame(rows), hide_index=True)
    else:
        st.success("No alerts on this day.")

# ============================ TAB 2: patterns ============================
with tab2:
    gl = raw["glucose_mgdl"]
    tir = ((gl >= LOW) & (gl <= HIGH)).mean() * 100
    k = st.columns(5)
    k[0].metric("Time in range 70-180", f"{tir:.0f}%")
    k[1].metric("Time above 180", f"{(gl > HIGH).mean() * 100:.0f}%")
    k[2].metric("Time below 70", f"{(gl < LOW).mean() * 100:.1f}%")
    k[3].metric("Average glucose", f"{gl.mean():.0f} mg/dL")
    k[4].metric("Variability (CV)", f"{gl.std() / gl.mean() * 100:.0f}%")

    st.markdown("##### Typical day (median and spread over 14 days)")
    slot = raw["timestamp"].dt.hour * 12 + raw["timestamp"].dt.minute // 5
    q = raw.assign(slot=slot).groupby("slot").glucose_mgdl.quantile([.1, .25, .5, .75, .9]).unstack()
    hrs = q.index.values * 5 / 60
    f2 = go.Figure()
    f2.add_trace(go.Scatter(x=hrs, y=q[0.9], line=dict(width=0), showlegend=False, hoverinfo="skip"))
    f2.add_trace(go.Scatter(x=hrs, y=q[0.1], fill="tonexty", fillcolor="rgba(59,130,246,0.15)",
                            line=dict(width=0), name="10th-90th percentile"))
    f2.add_trace(go.Scatter(x=hrs, y=q[0.75], line=dict(width=0), showlegend=False, hoverinfo="skip"))
    f2.add_trace(go.Scatter(x=hrs, y=q[0.25], fill="tonexty", fillcolor="rgba(59,130,246,0.30)",
                            line=dict(width=0), name="25th-75th percentile"))
    f2.add_trace(go.Scatter(x=hrs, y=q[0.5], line=dict(color=RED, width=3), name="Median",
                            hovertemplate="%{x:.1f} h  %{y:.0f} mg/dL<extra></extra>"))
    f2.add_hline(y=HIGH, line_dash="dash", line_color=ORANGE, line_width=1)
    f2.add_hline(y=LOW, line_dash="dash", line_color=BLUE, line_width=1)
    f2.update_layout(height=380, margin=dict(l=10, r=10, t=20, b=10), hovermode="x unified",
                     xaxis=dict(title="hour of day", dtick=3), yaxis=dict(title="glucose (mg/dL)"),
                     legend=dict(orientation="h", y=1.12))
    st.plotly_chart(f2, key="agp_chart")

    st.markdown("##### Every day at a glance (blue = low, green = in range, orange/red = high)")
    z = gl.values[: (len(gl) // 288) * 288].reshape(-1, 288)
    f3 = go.Figure(go.Heatmap(
        z=z, x=np.arange(288) * 5 / 60, y=[f"Day {i + 1}" for i in range(z.shape[0])],
        zmin=50, zmax=300, colorbar=dict(title="mg/dL"),
        colorscale=[[0, BLUE], [0.1, GREEN], [0.45, GREEN], [0.6, ORANGE], [1, RED]],
        hovertemplate="%{y}, %{x:.1f} h: %{z:.0f} mg/dL<extra></extra>"))
    f3.update_yaxes(autorange="reversed")
    f3.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(title="hour of day", dtick=3))
    st.plotly_chart(f3, key="heat_chart")

    st.markdown(f"##### Day {day} in detail: glucose, meals, activity, heart rate and sleep")
    d = raw.iloc[(day - 1) * 288: day * 288]
    dm = meals[(meals.patient_id == pid) & (meals.timestamp >= d.timestamp.iloc[0])
               & (meals.timestamp <= d.timestamp.iloc[-1])]
    k = st.columns(5)
    k[0].metric("Steps", f"{int(d.steps.sum()):,}")
    k[1].metric("Sleep", f"{d.sleeping.sum() * 5 / 60:.1f} h")
    k[2].metric("Average heart rate", f"{d.heart_rate.mean():.0f} bpm")
    k[3].metric("Average HRV", f"{d.hrv_ms.mean():.0f} ms")
    k[4].metric("Meals / walks after meal", f"{len(dm)} / {int(dm.walked_after.sum())}")

    f4 = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.5, 0.25, 0.25], vertical_spacing=0.04)
    f4.add_trace(go.Scatter(x=d.timestamp, y=d.glucose_mgdl, name="Glucose", line=dict(color=GRAY, width=2.5)), row=1, col=1)
    at = d.set_index("timestamp").glucose_mgdl.reindex(dm.timestamp).values
    f4.add_trace(go.Scatter(
        x=dm.timestamp, y=at, mode="markers", name="Meal",
        marker=dict(color=ORANGE, size=11, symbol="diamond", line=dict(color="white", width=1)),
        text=[f"{r.meal} ({int(r.carbs_g)} g carbs, GI {int(r.gi)}{', walked' if r.walked_after else ''})"
              for r in dm.itertuples()],
        hovertemplate="%{text}<extra></extra>"), row=1, col=1)
    f4.add_trace(go.Bar(x=d.timestamp, y=d.steps, name="Steps", marker_color=GREEN), row=2, col=1)
    f4.add_trace(go.Scatter(x=d.timestamp, y=d.heart_rate, name="Heart rate", line=dict(color=RED, width=1.8)), row=3, col=1)
    f4.add_hline(y=HIGH, line_dash="dash", line_color=ORANGE, line_width=1, row=1, col=1)
    f4.add_hline(y=LOW, line_dash="dash", line_color=BLUE, line_width=1, row=1, col=1)
    sl = d.sleeping.values
    edges = np.flatnonzero(np.diff(np.r_[0, sl, 0]))
    for a, b in zip(edges[::2], edges[1::2]):
        f4.add_vrect(x0=d.timestamp.iloc[a], x1=d.timestamp.iloc[b - 1], fillcolor=BLUE, opacity=0.10,
                     line_width=0, row="all", col=1)
    f4.update_layout(height=620, margin=dict(l=10, r=10, t=10, b=10), hovermode="x unified",
                     legend=dict(orientation="h", y=1.05))
    f4.update_yaxes(title_text="mg/dL", row=1, col=1)
    f4.update_yaxes(title_text="steps", row=2, col=1)
    f4.update_yaxes(title_text="bpm", row=3, col=1)
    st.plotly_chart(f4, key="day_chart")
    st.caption("Blue shading = sleep. Orange diamonds = meals.")

# ============================ TAB 3: what-if ============================
with tab3:
    st.markdown("##### What if the patient had done something different after a meal?")
    m = meals[(meals.patient_id == pid) & (meals.walked_after == 0)].copy()
    m["t0"] = ((m.timestamp - START).dt.total_seconds() / 300).round().astype(int)
    m = m[(m.t0 >= 1) & (m.t0 + AFTER >= HISTORY) & (m.t0 + AFTER + HORIZON <= n_steps)]
    m_day = m[(m.t0 >= (day - 1) * 288) & (m.t0 < day * 288)]
    if len(m_day) > 0:
        m = m_day
    m = m.reset_index(drop=True)
    if len(m) == 0:
        st.info("No meals available for this patient.")
    else:
        labs = [f"{(START + pd.Timedelta(minutes=5 * int(r.t0))).strftime('%d %b %H:%M')} - {r.meal} "
                f"({int(r.carbs_g)} g carbs)" for r in m.itertuples()]
        c1, c2 = st.columns(2)
        pick = c1.selectbox("Meal (the biggest meal of the chosen day is preselected)", labs,
                            index=int(m.carbs_g.values.argmax()))
        remedy = c2.selectbox("Remedy to highlight", REMEDIES, index=3)
        r = m.iloc[labs.index(pick)]
        t0, gi = int(r.t0), float(r.gi)
        i0 = t0 + AFTER
        w0 = x[i0 - HISTORY:i0][None].astype(np.float32)
        f0 = forecast(model, w0, s)[0]
        fx = ts[i0:i0 + HORIZON]

        f5 = go.Figure()
        f5.add_trace(go.Scatter(x=ts[i0 - 24:i0], y=g[i0 - 24:i0], name="Glucose so far",
                                line=dict(color=GRAY, width=3)))
        f5.add_trace(go.Scatter(x=fx, y=f0, name="No change", line=dict(color=GRAY, width=3, dash="dot")))
        results = []
        palette = {"walk 10 min": GREEN, "30% smaller portion": BLUE,
                   "low-GI swap (GI 50)": ORANGE, "walk + smaller portion": RED}
        for rem in REMEDIES:
            var = make_variant(x, t0, gi, rem)
            if var is None:
                continue
            f1 = forecast(model, var[None].astype(np.float32), s)[0]
            f5.add_trace(go.Scatter(x=fx, y=f1, name=rem,
                                    line=dict(color=palette[rem], width=4 if rem == remedy else 2)))
            results.append({"Remedy": rem, "Predicted peak": round(float(f1.max())),
                            "Change in peak": round(float(f1.max() - f0.max())),
                            "Minutes above 180 saved": int(((f0 > HIGH).sum() - (f1 > HIGH).sum()) * 5)})
            if rem == remedy:
                kk = st.columns(3)
                kk[0].metric("Predicted peak, no change", f"{f0.max():.0f} mg/dL")
                kk[1].metric(f"Peak with {rem}", f"{f1.max():.0f} mg/dL", f"{f1.max() - f0.max():.0f} mg/dL")
                kk[2].metric("Minutes above 180 saved", f"{((f0 > HIGH).sum() - (f1 > HIGH).sum()) * 5}")
        if remedy not in [x_["Remedy"] for x_ in results]:
            st.info("This meal already has a low glycemic index, so the swap changes nothing.")
        f5.add_hline(y=HIGH, line_dash="dash", line_color=ORANGE, line_width=1)
        f5.update_layout(height=430, margin=dict(l=10, r=10, t=20, b=10), hovermode="x unified",
                         legend=dict(orientation="h", y=1.12), yaxis=dict(title="glucose (mg/dL)"),
                         xaxis=dict(tickformat="%H:%M"))
        st.plotly_chart(f5, key="whatif_chart")
        st.caption("Click a name in the legend to hide or show that line.")
        st.dataframe(pd.DataFrame(results), hide_index=True)
        st.info("How far to trust this: checked against the simulator's true answers on unseen patients, "
                "portion-size and low-GI predictions were within about 1 mg/dL on average. The walking effect "
                "is underestimated (about 4 vs a true 13 mg/dL), so treat walk results as conservative.")

# ============================ TAB 4: record ============================
with tab4:
    st.markdown("##### Electronic health record (synthetic)")

    def rng(v, lo=None, hi=None):
        if hi is not None and v > hi:
            return "Above range"
        if lo is not None and v < lo:
            return "Below range"
        return "In range"

    rec = [
        ("Age", f"{row.age}", "", ""),
        ("Sex", f"{row.sex}", "", ""),
        ("BMI", f"{row.bmi}", "below 23 (Asian cut-off)", rng(row.bmi, hi=22.9)),
        ("Years since diagnosis", f"{row.years_since_diagnosis}", "", ""),
        ("HbA1c (%)", f"{row.hba1c}", "below 7.0 (typical target)", rng(row.hba1c, hi=6.99)),
        ("Fasting glucose (mg/dL)", f"{row.fasting_glucose}", "80 to 130", rng(row.fasting_glucose, 80, 130)),
        ("LDL (mg/dL)", f"{row.ldl}", "below 100", rng(row.ldl, hi=99)),
        ("Triglycerides (mg/dL)", f"{row.triglycerides}", "below 150", rng(row.triglycerides, hi=149)),
        ("Family history", "yes" if row.family_history else "no", "", ""),
        ("Genetic risk score (0-1)", f"{row.polygenic_risk}", "", ""),
        ("Medicines", ", ".join(meds) if meds else "none", "", ""),
    ]
    st.dataframe(pd.DataFrame(rec, columns=["Parameter", "Value", "Reference", "Flag"]), hide_index=True)
    st.caption("Flags compare values with common reference ranges. They are not a diagnosis.")

    with st.expander("How this digital twin works"):
        st.markdown(
            "- **Two data streams are fused.** The static record above (age, labs, medicines, genetic risk) "
            "and the dynamic sensor stream (glucose, steps, heart rate, HRV, sleep, meals).\n"
            "- **A neural network** reads the last 6 hours of sensor data together with the record, and "
            "forecasts glucose for the next 2 hours.\n"
            "- **Alerts** are raised when the forecast crosses 180 mg/dL (high) or nears 70 mg/dL (low).\n"
            "- **What-if remedies** re-run the forecast with one change (a walk, a smaller portion, a "
            "low-GI dish).\n"
            "- **All data is synthetic**, and these patients were never used for training."
        )

st.divider()
st.caption("Prototype for research and demonstration only. Not for clinical use.")
