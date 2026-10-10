"""
Doctor dashboard v3 for the Type 2 Diabetes digital twin (100% synthetic data).
Dark theme, scroll-linked animation, doctor's notes, patient avatar.

Run:  streamlit run app_v3.py
"""
import os
import json
import datetime
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
NOTES_FILE = "notes/doctor_notes.json"

st.set_page_config(page_title="T2D Digital Twin", page_icon="🩺", layout="wide")

CSS = """
<style>
html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    background: #000 !important; scroll-behavior: smooth; }
[data-testid="stHeader"] { background: rgba(0,0,0,.65); }
[data-testid="stSidebar"] { background: #07080b !important; border-right: 1px solid #1b1e26; }
.block-container { padding-top: 3.2rem; padding-bottom: 2rem; max-width: 1500px; }
[data-testid="stVerticalBlock"] { gap: .7rem; }
div[data-testid="stMetric"] { background: #0d0f14; border: 1px solid #1d212b; border-radius: 12px; padding: 10px 14px; }
.sec-title { font-size: 1.5rem; font-weight: 700; margin-top: 1.2rem; border-left: 4px solid #e4572e; padding-left: 12px; }
.sumcard { background: linear-gradient(90deg, #14171f, #0b0d12); border: 1px solid #262b36; border-left: 4px solid #3b82f6;
    border-radius: 12px; padding: 12px 18px; font-size: .95rem; }
.sumcard.warn { border-left-color: #f59e0b; }
.sumcard.good { border-left-color: #2ca02c; }
.avatar-card { background: #0d0f14; border: 1px solid #1d212b; border-radius: 16px; padding: 12px; text-align: center; }
.avatar-card .nm { font-weight: 700; margin-top: 6px; }
.avatar-card .sm { color: #9ca3af; font-size: .75rem; }
.floatbox { position: fixed; top: 62px; right: 18px; z-index: 99; background: rgba(13,15,20,.93); border: 1px solid #2a2f3b;
    border-radius: 12px; padding: 8px 14px; font-size: .8rem; line-height: 1.5; box-shadow: 0 6px 20px rgba(0,0,0,.6); }
.badge { display: inline-block; padding: 3px 10px; margin: 2px 6px 2px 0; border-radius: 999px; font-size: .82rem; }
.note { background: #0d0f14; border: 1px solid #262b36; border-radius: 10px; padding: 8px 10px; margin-bottom: 8px; font-size: .85rem; }
.note .meta { color: #9ca3af; font-size: .72rem; }
.navlink { display: block; color: #cbd5e1 !important; text-decoration: none; padding: 4px 0; }
.navlink:hover { color: #e4572e !important; }

@supports (animation-timeline: view()) {
  @keyframes rise { from { opacity: 0; transform: translateY(70px) scale(.97); } to { opacity: 1; transform: none; } }
  @keyframes fromleft { from { opacity: 0; transform: translateX(-100px); } to { opacity: 1; transform: none; } }
  div[data-testid="stMetric"], div[data-testid="stPlotlyChart"], .stPlotlyChart, div[data-testid="stDataFrame"], .avatar-card {
      animation: rise linear both; animation-timeline: view(); animation-range: entry 0% entry 90%; }
  .sumcard, .sec-title { animation: fromleft linear both; animation-timeline: view(); animation-range: entry 0% entry 100%; }
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


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
    if len(ms) == 0:
        return np.array([], dtype=int), np.empty((0,), dtype=np.float32)
    W = np.stack([x[m - HISTORY:m] for m in ms]).astype(np.float32)
    S = np.repeat(s[None], len(ms), axis=0)
    with torch.no_grad():
        pred = model(torch.from_numpy(W), torch.from_numpy(S)).numpy()
    now = g[np.array(ms) - 1]
    return ms, now[:, None] + pred * 100.0


# ----------------------------- helpers -----------------------------
def load_notes():
    try:
        with open(NOTES_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_notes(d):
    os.makedirs("notes", exist_ok=True)
    with open(NOTES_FILE, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=2)


def dark(fig, h, **kw):
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      height=h, **kw)
    return fig


def badge(text, color):
    return f'<span class="badge" style="background:{color}22;border:1px solid {color};color:{color}">{text}</span>'


def sec(anchor, icon, title):
    st.markdown(f'<div id="{anchor}"></div><div class="sec-title">{icon} {title}</div>', unsafe_allow_html=True)


def card(html, kind=""):
    return f'<div class="sumcard {kind}">{html}</div>'


def avatar_svg(pid, sex, age):
    seed = sum(map(ord, pid))
    skin = ["#f1c27d", "#e0ac69", "#c68642", "#8d5524", "#d9a066"][seed % 5]
    bg = ["#1e3a5f", "#3b2f5f", "#1f4d3a", "#5f2f3b", "#4a4a1f"][(seed // 5) % 5]
    hair = "#d9d9d9" if age >= 60 else ["#111111", "#2b1b10", "#3a2a1a"][seed % 3]
    parts = [
        '<svg viewBox="0 0 120 120" width="120" height="120" xmlns="http://www.w3.org/2000/svg">',
        f'<rect width="120" height="120" rx="16" fill="{bg}"/>',
        '<path d="M18 120 Q18 84 60 84 Q102 84 102 120Z" fill="#3b82f6"/>',
    ]
    if sex == "F":
        parts.append(f'<rect x="31" y="40" width="14" height="46" rx="7" fill="{hair}"/>')
        parts.append(f'<rect x="75" y="40" width="14" height="46" rx="7" fill="{hair}"/>')
    parts += [
        f'<circle cx="60" cy="54" r="24" fill="{skin}"/>',
        f'<path d="M35 52 Q35 26 60 26 Q85 26 85 52 Q74 38 60 38 Q46 38 35 52Z" fill="{hair}"/>',
        '<circle cx="51" cy="56" r="2.6" fill="#111"/><circle cx="69" cy="56" r="2.6" fill="#111"/>',
        '<path d="M52 66 Q60 72 68 66" stroke="#111" stroke-width="2.4" fill="none" stroke-linecap="round"/>',
        '</svg>',
    ]
    return "".join(parts)


# ----------------------------- sidebar -----------------------------
with st.sidebar:
    st.markdown("### Controls")
    pid = st.selectbox("Patient (unseen during training)", test_ids)
    day = st.slider("Day to review", 1, 14, 4)
    low_sens = st.radio("Low-glucose alert sensitivity", ["High (recommended)", "Standard"], index=0)
    st.caption("High sensitivity catches more lows but gives more false alarms. A missed low is the more dangerous mistake.")
    st.markdown('<a class="navlink" href="#sec-live">▸ Live twin</a><a class="navlink" href="#sec-patterns">▸ 14-day patterns</a>'
                '<a class="navlink" href="#sec-whatif">▸ What-if remedies</a><a class="navlink" href="#sec-record">▸ Patient record</a>',
                unsafe_allow_html=True)
    st.divider()
    st.markdown("### 📝 Doctor's notes")
    notes = load_notes()
    with st.form("note_form", clear_on_submit=True):
        tag = st.selectbox("Type", ["Important", "Follow-up", "Medication", "Lifestyle", "Other"])
        txt = st.text_area("Note", height=110, placeholder="Write what you want to remember about this patient...")
        saved = st.form_submit_button("Save note")
    if saved and txt.strip():
        notes.setdefault(pid, []).append({
            "time": datetime.datetime.now().strftime("%d %b %Y, %H:%M"),
            "tag": tag, "text": txt.strip(), "context": f"reviewing day {day}"})
        save_notes(notes)
        st.toast("Note saved")
    mine = notes.get(pid, [])
    st.caption(f"{len(mine)} note(s) for {pid}")
    for n in reversed(mine[-6:]):
        body = n["text"].replace("<", "&lt;").replace("\n", "<br>")
        st.markdown(f'<div class="note"><b>{n["tag"]}</b><div class="meta">{n["time"]} · {n["context"]}</div>{body}</div>',
                    unsafe_allow_html=True)

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
meds = [n for n, v in (("Metformin", row.metformin), ("Sulfonylurea", row.sulfonylurea), ("Insulin", row.insulin)) if v]

flags = []
if row.hba1c >= 7.0:
    flags.append(("HbA1c at or above 7%", ORANGE))
if row.ldl >= 100:
    flags.append(("LDL at or above 100", ORANGE))
if row.triglycerides >= 150:
    flags.append(("Triglycerides at or above 150", ORANGE))
if row.bmi >= 23:
    flags.append(("BMI at or above 23 (Asian overweight cut-off)", ORANGE))
if row.family_history:
    flags.append(("Family history of diabetes", BLUE))
if row.insulin or row.sulfonylurea:
    flags.append(("Medicine can cause low glucose", RED))

# ----------------------------- header -----------------------------
st.markdown("## 🩺 Type 2 Diabetes Digital Twin")
st.caption("Prototype on 100% synthetic data. Not medical advice. Needs clinical validation.")

hc1, hc2 = st.columns([1, 6])
with hc1:
    st.markdown(f'<div class="avatar-card">{avatar_svg(pid, row.sex, row.age)}<div class="nm">{pid}</div>'
                f'<div class="sm">{row.age} y, {"Male" if row.sex == "M" else "Female"}<br>Illustration, synthetic patient</div></div>',
                unsafe_allow_html=True)
with hc2:
    cols = st.columns(5)
    cols[0].metric("BMI", f"{row.bmi}")
    cols[1].metric("HbA1c", f"{row.hba1c}%")
    cols[2].metric("Fasting glucose", f"{row.fasting_glucose} mg/dL")
    cols[3].metric("Years since diagnosis", f"{row.years_since_diagnosis}")
    cols[4].metric("Genetic risk score", f"{row.polygenic_risk}")
    html = "".join(badge(m, GREEN) for m in meds) or badge("No diabetes medicine", GRAY)
    html += "".join(badge(t, c) for t, c in flags)
    st.markdown(html, unsafe_allow_html=True)

# ============================ SECTION 1: live twin ============================
sec("sec-live", "📈", "Live twin: 2-hour glucose forecast")
box_live = st.empty()

start = max((day - 1) * 288, HISTORY)
stop = min(day * 288, n_steps - HORIZON - 1)
ms, F = day_forecasts(pid, start, stop, 12)

if len(ms) == 0:
    box_live.markdown(card(
        f"<b>Day {day} summary:</b> no valid forecast window for this day. Try another day within the patient record.",
        "warn"), unsafe_allow_html=True)
    st.info("The selected day has no forecast points available; this usually happens when the patient record is shorter than the requested review window.")
else:
    alerts = [alert_of(g[m - 1], F[i]) for i, m in enumerate(ms)]
    sym = {"HIGH": " ▲", "LOW": " ▼", "OK": ""}
    labels = [ts[m - 1].strftime("%H:%M") + sym[a] for m, a in zip(ms, alerts)]
    n_high, n_low = alerts.count("HIGH"), alerts.count("LOW")
    kind = "warn" if (n_high or n_low) else "good"
    box_live.markdown(card(
        f"<b>Day {day} summary:</b> {n_high} high-glucose alert(s) ▲, {n_low} low-glucose alert(s) ▼ across {len(ms)} moments. "
        f"Highest forecast <b>{F.max():.0f} mg/dL</b>, lowest <b>{F.min():.0f} mg/dL</b>. Press ▶ Play to watch the twin move through the day.",
        kind), unsafe_allow_html=True)

    def traces_for(m, fc):
        a, b = m - HISTORY, m + HORIZON
        return [(ts[a:m], g[a:m]), (ts[m - 1:b], g[m - 1:b]), (ts[m - 1:b], np.r_[g[m - 1], fc]), (ts[m - 1:m], g[m - 1:m])]

    first = traces_for(ms[0], F[0])
    hov = "%{x|%H:%M}  %{y:.0f} mg/dL<extra></extra>"
    base = [
        go.Scatter(x=first[0][0], y=first[0][1], name="Glucose (last 6 h)", line=dict(color=GRAY, width=3), hovertemplate=hov),
        go.Scatter(x=first[1][0], y=first[1][1], name="What actually happened (hidden from model)",
                   line=dict(color=GRAY, width=2, dash="dot"), hovertemplate=hov),
        go.Scatter(x=first[2][0], y=first[2][1], name="2-hour forecast", line=dict(color=RED, width=3), hovertemplate=hov),
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
    fig.add_hrect(y0=LOW, y1=HIGH, fillcolor=GREEN, opacity=0.08, line_width=0)
    fig.add_hline(y=HIGH, line_dash="dash", line_color=ORANGE, line_width=1)
    fig.add_hline(y=LOW, line_dash="dash", line_color=BLUE, line_width=1)
    play_args = dict(frame=dict(duration=700, redraw=False), transition=dict(duration=500, easing="cubic-in-out"),
                     fromcurrent=True, mode="immediate")
    dark(fig, 520, margin=dict(l=10, r=10, t=30, b=120), hovermode="x unified", legend=dict(orientation="h", y=1.1),
         yaxis=dict(title="glucose (mg/dL)", range=[lo, hi]),
         xaxis=dict(range=[str(ts[ms[0] - HISTORY]), str(ts[ms[0] + HORIZON - 1])], tickformat="%H:%M<br>%d %b"),
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
                              for i in range(len(ms))])])

    lc1, lc2 = st.columns([3, 1])
    with lc1:
        st.plotly_chart(fig, key="live_chart")
        st.caption("▲ high-glucose alert, ▼ low-glucose alert. The dotted line is the real glucose that followed; the model never sees it.")
    with lc2:
        rows = [{"Time": ts[m - 1].strftime("%H:%M"), "Peak": round(float(F[i].max())), "Low": round(float(F[i].min())),
                 "Alert": alerts[i]} for i, m in enumerate(ms) if alerts[i] != "OK"]
        if rows:
            st.markdown("**Alerts raised this day**")
            st.dataframe(pd.DataFrame(rows), hide_index=True, height=480)
        else:
            st.success("No alerts on this day.")

# ============================ SECTION 2: patterns ============================
sec("sec-patterns", "🗓️", "14-day patterns")
box_pat = st.empty()
gl = raw["glucose_mgdl"]
tir = ((gl >= LOW) & (gl <= HIGH)).mean() * 100
above = (gl > HIGH).mean() * 100
below = (gl < LOW).mean() * 100
cv = gl.std() / gl.mean() * 100
box_pat.markdown(card(
    f"<b>Overall:</b> {tir:.0f}% of the time in range 70-180 (a commonly used goal is above 70%), {above:.0f}% above 180, "
    f"{below:.1f}% below 70. Average glucose {gl.mean():.0f} mg/dL, variability {cv:.0f}%.",
    "good" if tir >= 70 else "warn"), unsafe_allow_html=True)

k = st.columns(5)
k[0].metric("Time in range 70-180", f"{tir:.0f}%")
k[1].metric("Time above 180", f"{above:.0f}%")
k[2].metric("Time below 70", f"{below:.1f}%")
k[3].metric("Average glucose", f"{gl.mean():.0f} mg/dL")
k[4].metric("Variability (CV)", f"{cv:.0f}%")

pc1, pc2 = st.columns(2)
with pc1:
    st.markdown("**Typical day** (median and spread over 14 days)")
    slot = raw["timestamp"].dt.hour * 12 + raw["timestamp"].dt.minute // 5
    q = raw.assign(slot=slot).groupby("slot").glucose_mgdl.quantile([.1, .25, .5, .75, .9]).unstack()
    hrs = q.index.values * 5 / 60
    f2 = go.Figure()
    f2.add_trace(go.Scatter(x=hrs, y=q[0.9], line=dict(width=0), showlegend=False, hoverinfo="skip"))
    f2.add_trace(go.Scatter(x=hrs, y=q[0.1], fill="tonexty", fillcolor="rgba(59,130,246,0.18)",
                            line=dict(width=0), name="10th-90th percentile"))
    f2.add_trace(go.Scatter(x=hrs, y=q[0.75], line=dict(width=0), showlegend=False, hoverinfo="skip"))
    f2.add_trace(go.Scatter(x=hrs, y=q[0.25], fill="tonexty", fillcolor="rgba(59,130,246,0.35)",
                            line=dict(width=0), name="25th-75th percentile"))
    f2.add_trace(go.Scatter(x=hrs, y=q[0.5], line=dict(color=RED, width=3), name="Median",
                            hovertemplate="%{x:.1f} h  %{y:.0f} mg/dL<extra></extra>"))
    f2.add_hline(y=HIGH, line_dash="dash", line_color=ORANGE, line_width=1)
    f2.add_hline(y=LOW, line_dash="dash", line_color=BLUE, line_width=1)
    dark(f2, 360, margin=dict(l=10, r=10, t=10, b=10), hovermode="x unified",
         xaxis=dict(title="hour of day", dtick=3), yaxis=dict(title="glucose (mg/dL)"),
         legend=dict(orientation="h", y=1.15))
    st.plotly_chart(f2, key="agp_chart")
with pc2:
    st.markdown("**Every day at a glance** (blue low, green in range, orange/red high)")
    z = gl.values[: (len(gl) // 288) * 288].reshape(-1, 288)
    f3 = go.Figure(go.Heatmap(
        z=z, x=np.arange(288) * 5 / 60, y=[f"Day {i + 1}" for i in range(z.shape[0])],
        zmin=50, zmax=300, colorbar=dict(title="mg/dL"),
        colorscale=[[0, BLUE], [0.1, GREEN], [0.45, GREEN], [0.6, ORANGE], [1, RED]],
        hovertemplate="%{y}, %{x:.1f} h: %{z:.0f} mg/dL<extra></extra>"))
    f3.update_yaxes(autorange="reversed")
    dark(f3, 360, margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(title="hour of day", dtick=3))
    st.plotly_chart(f3, key="heat_chart")

st.markdown(f"**Day {day} in detail:** glucose, meals, activity, heart rate and sleep")
d = raw.iloc[(day - 1) * 288: day * 288]
dm = meals[(meals.patient_id == pid) & (meals.timestamp >= d.timestamp.iloc[0]) & (meals.timestamp <= d.timestamp.iloc[-1])]
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
    text=[f"{r.meal} ({int(r.carbs_g)} g carbs, GI {int(r.gi)}{', walked' if r.walked_after else ''})" for r in dm.itertuples()],
    hovertemplate="%{text}<extra></extra>"), row=1, col=1)
f4.add_trace(go.Bar(x=d.timestamp, y=d.steps, name="Steps", marker_color=GREEN), row=2, col=1)
f4.add_trace(go.Scatter(x=d.timestamp, y=d.heart_rate, name="Heart rate", line=dict(color=RED, width=1.8)), row=3, col=1)
f4.add_hline(y=HIGH, line_dash="dash", line_color=ORANGE, line_width=1, row=1, col=1)
f4.add_hline(y=LOW, line_dash="dash", line_color=BLUE, line_width=1, row=1, col=1)
sl = d.sleeping.values
edges = np.flatnonzero(np.diff(np.r_[0, sl, 0]))
for a_, b_ in zip(edges[::2], edges[1::2]):
    f4.add_vrect(x0=d.timestamp.iloc[a_], x1=d.timestamp.iloc[b_ - 1], fillcolor=BLUE, opacity=0.14,
                 line_width=0, row="all", col=1)
dark(f4, 560, margin=dict(l=10, r=10, t=10, b=10), hovermode="x unified", legend=dict(orientation="h", y=1.05))
f4.update_yaxes(title_text="mg/dL", row=1, col=1)
f4.update_yaxes(title_text="steps", row=2, col=1)
f4.update_yaxes(title_text="bpm", row=3, col=1)
st.plotly_chart(f4, key="day_chart")
st.caption("Blue shading = sleep. Orange diamonds = meals.")

# ============================ SECTION 3: what-if ============================
sec("sec-whatif", "🧪", "What-if remedies")
box_wi = st.empty()
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
    labs = [f"{(START + pd.Timedelta(minutes=5 * int(r.t0))).strftime('%d %b %H:%M')} - {r.meal} ({int(r.carbs_g)} g carbs)"
            for r in m.itertuples()]
    c1, c2 = st.columns(2)
    pick = c1.selectbox("Meal (the biggest meal of the chosen day is preselected)", labs, index=int(m.carbs_g.values.argmax()))
    remedy = c2.selectbox("Remedy to highlight", REMEDIES, index=3)
    r = m.iloc[labs.index(pick)]
    t0, gi = int(r.t0), float(r.gi)
    i0 = t0 + AFTER
    f0 = forecast(model, x[i0 - HISTORY:i0][None].astype(np.float32), s)[0]
    fx = ts[i0:i0 + HORIZON]

    f5 = go.Figure()
    f5.add_trace(go.Scatter(x=ts[i0 - 24:i0], y=g[i0 - 24:i0], name="Glucose so far", line=dict(color=GRAY, width=3)))
    f5.add_trace(go.Scatter(x=fx, y=f0, name="No change", line=dict(color=GRAY, width=3, dash="dot")))
    results = []
    palette = {"walk 10 min": GREEN, "30% smaller portion": BLUE, "low-GI swap (GI 50)": ORANGE, "walk + smaller portion": RED}
    shown = None
    for rem in REMEDIES:
        var = make_variant(x, t0, gi, rem)
        if var is None:
            continue
        f1 = forecast(model, var[None].astype(np.float32), s)[0]
        f5.add_trace(go.Scatter(x=fx, y=f1, name=rem, line=dict(color=palette[rem], width=4 if rem == remedy else 2)))
        results.append({"Remedy": rem, "Predicted peak": round(float(f1.max())),
                        "Change in peak": round(float(f1.max() - f0.max())),
                        "Min above 180 saved": int(((f0 > HIGH).sum() - (f1 > HIGH).sum()) * 5)})
        if rem == remedy:
            shown = (f1.max(), ((f0 > HIGH).sum() - (f1 > HIGH).sum()) * 5)
    f5.add_hline(y=HIGH, line_dash="dash", line_color=ORANGE, line_width=1)
    dark(f5, 400, margin=dict(l=10, r=10, t=10, b=10), hovermode="x unified", legend=dict(orientation="h", y=1.12),
         yaxis=dict(title="glucose (mg/dL)"), xaxis=dict(tickformat="%H:%M"))

    if results:
        best = min(results, key=lambda z_: z_["Change in peak"])
        box_wi.markdown(card(
            f"<b>For this meal</b> (predicted peak without any change: <b>{f0.max():.0f} mg/dL</b>): the most helpful option the model "
            f"predicts is <b>{best['Remedy']}</b>, changing the peak by <b>{best['Change in peak']} mg/dL</b>. "
            f"These are model predictions checked only against the simulator, not real patients.", "warn"), unsafe_allow_html=True)
    wc1, wc2 = st.columns([3, 2])
    with wc1:
        st.plotly_chart(f5, key="whatif_chart")
        st.caption("Click a name in the legend to hide or show that line.")
    with wc2:
        if shown is not None:
            kk = st.columns(2)
            kk[0].metric("Predicted peak, no change", f"{f0.max():.0f} mg/dL")
            kk[1].metric(f"Peak with {remedy}", f"{shown[0]:.0f} mg/dL", f"{shown[0] - f0.max():.0f} mg/dL")
        else:
            st.info("This meal already has a low glycemic index, so the swap changes nothing.")
        st.dataframe(pd.DataFrame(results), hide_index=True)
        st.info("Checked against the simulator's true answers on unseen patients: portion-size and low-GI predictions were "
                "within about 1 mg/dL on average. The walking effect was also close (14.9 vs a true 13.2 mg/dL).")

# ============================ SECTION 4: record ============================
sec("sec-record", "📋", "Patient record")
box_rec = st.empty()
box_rec.markdown(card(
    f"<b>{len(flags)} flag(s):</b> " + ("; ".join(t for t, _ in flags) if flags else "none") +
    f". Medicines: {', '.join(meds) if meds else 'none'}.", "warn" if flags else "good"), unsafe_allow_html=True)


def rng(v, lo_=None, hi_=None):
    if hi_ is not None and v > hi_:
        return "Above range"
    if lo_ is not None and v < lo_:
        return "Below range"
    return "In range"


rec = [
    ("Age", f"{row.age}", "", ""),
    ("Sex", f"{row.sex}", "", ""),
    ("BMI", f"{row.bmi}", "below 23 (Asian cut-off)", rng(row.bmi, hi_=22.9)),
    ("Years since diagnosis", f"{row.years_since_diagnosis}", "", ""),
    ("HbA1c (%)", f"{row.hba1c}", "below 7.0 (typical target)", rng(row.hba1c, hi_=6.99)),
    ("Fasting glucose (mg/dL)", f"{row.fasting_glucose}", "80 to 130", rng(row.fasting_glucose, 80, 130)),
    ("LDL (mg/dL)", f"{row.ldl}", "below 100", rng(row.ldl, hi_=99)),
    ("Triglycerides (mg/dL)", f"{row.triglycerides}", "below 150", rng(row.triglycerides, hi_=149)),
    ("Family history", "yes" if row.family_history else "no", "", ""),
    ("Genetic risk score (0-1)", f"{row.polygenic_risk}", "", ""),
    ("Medicines", ", ".join(meds) if meds else "none", "", ""),
]
rc1, rc2 = st.columns([3, 2])
with rc1:
    st.dataframe(pd.DataFrame(rec, columns=["Parameter", "Value", "Reference", "Flag"]), hide_index=True, height=430)
    st.caption("Flags compare values with common reference ranges. They are not a diagnosis.")
with rc2:
    st.markdown("**How this digital twin works**")
    st.markdown(
        "- **Two data streams are fused:** the static record (age, labs, medicines, genetic risk) and the sensor stream "
        "(glucose, steps, heart rate, HRV, sleep, meals).\n"
        "- **A neural network** reads the last 6 hours of sensor data with the record and forecasts the next 2 hours.\n"
        "- **Alerts** fire when the forecast crosses 180 mg/dL (high) or nears 70 mg/dL (low).\n"
        "- **What-if remedies** re-run the forecast with one change.\n"
        "- **All data is synthetic**, and these patients were never used for training.")

st.caption("Prototype for research and demonstration only. Not for clinical use.")

# floating summary box (stays visible while scrolling)
st.markdown(f'<div class="floatbox"><b>{pid}</b> · {row.age} y {row.sex}<br>HbA1c {row.hba1c}% · Day {day}<br>'
            f'▲ {n_high} high · ▼ {n_low} low alerts</div>', unsafe_allow_html=True)
