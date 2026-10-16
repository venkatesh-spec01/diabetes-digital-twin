# streamlit panels for the doctor dashboard:
# the glucose level bar, safety view with uncertainty band, reference actions,
# medicine table, diet chart window and the patient summary with pdf.
# the numbers come from doctor_logic.py and the lists from clinical_reference.py.
import datetime
import html

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import clinical_reference as cr
import doctor_logic as dl

# colours used in the charts
GREEN, RED, BLUE, ORANGE, GRAY = "#2ca02c", "#e4572e", "#3b82f6", "#f59e0b", "#9ca3af"
# colour for each alert type
KIND_COLOUR = {"HIGH": ORANGE, "LOW": BLUE, "OK": GREEN}

# small styling for tables and status card
CSS = """
<style>
.rtable { width: 100%; border-collapse: collapse; font-size: .85rem; margin-bottom: .6rem; }
.rtable th { background: #14171f; color: #cbd5e1; text-align: left; padding: 6px 8px; border: 1px solid #262b36; }
.rtable td { vertical-align: top; padding: 6px 8px; border: 1px solid #262b36; }
.statuscard { border: 1px solid #262b36; border-left: 5px solid #3b82f6; border-radius: 12px; padding: 12px 18px;
    background: linear-gradient(90deg, #14171f, #0b0d12); font-size: .95rem; }
</style>
"""


# adds the styling to the page
def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


# turns a table into html
def html_table(df):
    return df.to_html(index=False, classes="rtable", border=0, escape=True)


# dark theme for plotly charts
def _dark(fig, h, **kw):
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=h, **kw)
    return fig


# colour bar showing where glucose sits
def level_bar(now, fc):
    fig = go.Figure()
    xmin, xmax = 40, 320
    # draw one coloured block for each zone
    for lo, hi, name, colour in cr.ZONES:
        x0, x1 = max(lo, xmin), min(hi, xmax)
        if x1 <= x0:
            continue
        fig.add_shape(type="rect", x0=x0, x1=x1, y0=0, y1=1, fillcolor=colour, opacity=0.35, line_width=0, layer="below")
        fig.add_annotation(x=(x0 + x1) / 2, y=1.18, text=name, showarrow=False, font=dict(size=11, color=colour))
    # forecast peak and lowest points as markers
    peak, low = float(np.max(fc)), float(np.min(fc))
    fig.add_trace(go.Scatter(x=[min(max(low, xmin), xmax)], y=[0.5], mode="markers+text", text=[f"lowest {low:.0f}"],
                             textposition="bottom center", name="Forecast lowest",
                             marker=dict(symbol="triangle-down", size=16, color="white", line=dict(color="#111", width=1))))
    fig.add_trace(go.Scatter(x=[min(max(peak, xmin), xmax)], y=[0.5], mode="markers+text", text=[f"peak {peak:.0f}"],
                             textposition="top center", name="Forecast peak",
                             marker=dict(symbol="triangle-up", size=16, color="white", line=dict(color="#111", width=1))))
    fig.add_trace(go.Scatter(x=[min(max(now, xmin), xmax)], y=[0.5], mode="markers+text", text=[f"now {now:.0f}"],
                             textposition="middle right", name="Now",
                             marker=dict(symbol="circle", size=20, color=RED, line=dict(color="white", width=2))))
    _dark(fig, 190, margin=dict(l=10, r=10, t=30, b=10), showlegend=False,
          xaxis=dict(range=[xmin, xmax], title="glucose (mg/dL)"),
          yaxis=dict(visible=False, range=[-0.2, 1.45]))
    return fig


# shows the current value and the level bar
def render_level(now, fc, key):
    name, colour = dl.zone_of(now)
    c1, c2 = st.columns([1, 3])
    with c1:
        st.metric("Glucose now", f"{now:.0f} mg/dL")
        st.markdown(f'<span style="color:{colour};font-weight:700">{name}</span>', unsafe_allow_html=True)
        st.caption(f"Forecast: lowest {np.min(fc):.0f}, peak {np.max(fc):.0f} mg/dL over the next 2 hours")
    with c2:
        st.plotly_chart(level_bar(now, fc), key=key)


# shows status card and forecast with band
def render_safety(ts, g, m, fc, sigma, status, key, horizon=24, history=72):
    colour = KIND_COLOUR[status["kind"]] if status["label"] != "Clear" else GREEN
    badge = f'<b style="color:{colour}">{html.escape(status["label"])}</b>'
    st.markdown(f'<div class="statuscard" style="border-left-color:{colour}">{badge} &nbsp;|&nbsp; '
                f'{html.escape(status["confidence"])} &nbsp;|&nbsp; {html.escape(status["reason"])}</div>', unsafe_allow_html=True)
    # upper and lower edge of the likely range
    lo_b, hi_b = dl.band(fc, sigma)
    xs = ts[m - 1:m + horizon]
    now = float(g[m - 1])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ts[m - history:m], y=g[m - history:m], name="Glucose (last 6 h)", line=dict(color=GRAY, width=3)))
    fig.add_trace(go.Scatter(x=xs, y=np.r_[now, hi_b], line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=xs, y=np.r_[now, lo_b], fill="tonexty", fillcolor="rgba(228,87,46,0.20)", line=dict(width=0),
                             name="Likely range (about 80%)"))
    fig.add_trace(go.Scatter(x=xs, y=np.r_[now, fc], name="2-hour forecast", line=dict(color=RED, width=3)))
    fig.add_hline(y=180, line_dash="dash", line_color=ORANGE, line_width=1)
    fig.add_hline(y=70, line_dash="dash", line_color=BLUE, line_width=1)
    _dark(fig, 360, margin=dict(l=10, r=10, t=10, b=10), hovermode="x unified", legend=dict(orientation="h", y=1.12),
          yaxis=dict(title="glucose (mg/dL)"), xaxis=dict(tickformat="%H:%M"))
    st.plotly_chart(fig, key=key)
    st.caption("The shaded band is built from this patient's own forecast errors over the whole record (about 80% of "
               "past errors fell inside it). A forecast whose band crosses 180 or 70 is marked as a possible alert, so a "
               "borderline case is not silently missed. This is a simple estimate, not a validated confidence interval.")


# shows the action list for high or low
def render_actions(kind, possible, key):
    if kind == "OK":
        st.success("No action is needed for this moment. The forecast and its band stay inside 70-180 mg/dL.")
        return
    rows = cr.ACTIONS_HIGH if kind == "HIGH" else cr.ACTIONS_LOW
    df = pd.DataFrame(rows, columns=cr.ACTION_COLUMNS)
    # adds a timing group for each action
    df["Timing"] = df["When to do it"].map(cr.timing_group)
    title = "high glucose" if kind == "HIGH" else "low glucose"
    st.markdown(f"**Reference actions for {title}**" + (" (possible, borderline forecast)" if possible else ""))
    st.warning("General reference for the treating doctor, not a prescription. No doses are calculated here. "
               "The hospital protocol and the doctor's judgement come first.")
    c1, c2 = st.columns(2)
    group_by = c1.radio("Group by", ["Type of action", "When to do it (before / with / after the meal)"], key=f"{key}_grp")
    types = c2.multiselect("Show types", sorted(df["Type"].unique()), default=sorted(df["Type"].unique()), key=f"{key}_types")
    df = df[df["Type"].isin(types)]
    if df.empty:
        st.info("Nothing selected.")
        return
    col = "Type" if group_by.startswith("Type") else "Timing"
    order = list(df[col].unique()) if col == "Type" else [t for t in cr.TIMING_FILTER if t in set(df[col])]
    for grp in order:
        sub = df[df[col] == grp]
        st.markdown(f"**{grp}**")
        st.markdown(html_table(sub[cr.ACTION_COLUMNS]), unsafe_allow_html=True)


# shows and edits the medicine table
def render_medicines(pid, data_dir):
    df, saved = dl.load_meds(pid, data_dir)
    n_bad = 0
    note = dl.adherence_note(df)
    # warning when the patient misses doses
    if note:
        n_bad = note.count(";") + 1
    k = st.columns(3)
    k[0].metric("Medicines on the hospital list", len(df))
    k[1].metric("Not taken regularly", n_bad)
    k[2].metric("List source", "Saved by doctor" if saved else "Synthetic sample")
    if note:
        st.warning(f"Missed doses reported: {note}. Check this before increasing any dose.")
    # medicines that can cause low glucose
    risky = df[df["Note"].astype(str).str.contains("low glucose", case=False)] if len(df) else df
    if len(risky):
        st.info("Medicine that can cause low glucose: " + ", ".join(risky["Medicine"]) + ". Keep the low-glucose alert on high sensitivity.")
    st.caption("Doses below are typical example doses on synthetic patients, not prescriptions. The doctor can edit the table "
               "(add or remove rows) and press Save. 'Patient reports' is what the patient says they actually take.")
    # doctor can edit this table directly
    edited = st.data_editor(
        df, num_rows="dynamic", hide_index=True, key=f"meds_editor_{pid}",
        column_config={"Patient reports": st.column_config.SelectboxColumn(
            "Patient reports", options=["Taking regularly", "Sometimes missed", "Often missed", "Stopped"])})
    b1, b2 = st.columns([1, 1])
    if b1.button("Save medicine list", key=f"meds_save_{pid}"):
        dl.save_meds(pid, edited)
        st.toast("Medicine list saved")
    if saved and b2.button("Reset to sample list", key=f"meds_reset_{pid}"):
        import os
        try:
            os.remove(os.path.join("notes", f"meds_{pid}.json"))
        except OSError:
            pass
        st.rerun()


# opens the diet chart in a pop-up window
@st.dialog("Diet chart given to the patient", width="large")
def _diet_dialog(pid, bmi, habit, on_risky_drug):
    default = cr.default_plan_for(bmi)
    saved_df, saved_plan, saved = dl.load_diet(pid, default)
    plans = list(cr.DIET_PLANS)
    start = saved_plan if saved_plan in plans else default
    plan = st.selectbox("Sample plan", plans, index=plans.index(start), key=f"diet_plan_{pid}")
    df = saved_df if (saved and plan == saved_plan) else dl.diet_table(plan)
    st.caption(f"Suggested from BMI {bmi}. Eating pattern in the record: {habit.replace('_', ' ')}. "
               "Sample Indian plan, to be personalised by a dietitian. Carbohydrate values are approximate.")
    if on_risky_drug:
        st.warning("This patient takes a medicine that can cause low glucose. Fixed meal times matter and meals should not be skipped.")
    # diet table can be edited too
    edited = st.data_editor(df, hide_index=True, num_rows="dynamic", key=f"diet_editor_{pid}_{plan}")
    # adds up carbs for the whole day
    total = pd.to_numeric(edited["Carbs (g, approx.)"], errors="coerce").sum()
    st.metric("Carbohydrate in the whole day (approx.)", f"{total:.0f} g")
    st.markdown("**Simple rules to give with the chart**")
    for r in cr.DIET_RULES:
        st.markdown(f"- {r}")
    c1, c2 = st.columns(2)
    if c1.button("Save changes", key=f"diet_save_{pid}"):
        dl.save_diet(pid, plan, edited)
        st.toast("Diet chart saved")
    c2.download_button("Download as CSV", edited.to_csv(index=False).encode("utf-8"), file_name=f"diet_chart_{pid}.csv",
                       mime="text/csv", key=f"diet_dl_{pid}")


# button that opens the diet chart
def diet_button(pid, bmi, habit, on_risky_drug):
    if st.button("📋 Open diet chart (new window)", key=f"diet_open_{pid}"):
        _diet_dialog(pid, float(bmi), str(habit), bool(on_risky_drug))


# shows summary numbers and the pdf button
def render_summary(pid, row, stats, meds_df, ts, gl, sigma2h, flags, diet_plan):
    k = st.columns(5)
    k[0].metric("Time in range 70-180", f"{stats['tir']:.0f}%")
    k[1].metric("Time above 180", f"{stats['above']:.0f}%")
    k[2].metric("Time below 70", f"{stats['below']:.1f}%")
    k[3].metric("High alerts (whole record)", stats["n_high"])
    k[4].metric("Low alerts (whole record)", stats["n_low"])
    st.caption(f"Alerts are counted over {stats['n_moments']} forecast moments across the whole record. "
               f"Typical forecast error at 2 hours for this patient: about {sigma2h:.0f} mg/dL.")
    # remember the pdf so the download works
    key = f"pdf_bytes_{pid}"
    # build the pdf only when button is clicked
    if st.button("Create one-page PDF summary", key=f"pdf_make_{pid}"):
        info = dict(pid=pid, age=int(row.age), sex=row.sex, bmi=row.bmi, hba1c=row.hba1c, fasting=row.fasting_glucose,
                    years=row.years_since_diagnosis, flags=flags, meds=meds_df, stats=stats, ts=ts, gl=gl,
                    sigma2h=float(sigma2h), diet_plan=diet_plan, today=datetime.date.today().strftime("%d %b %Y"))
        st.session_state[key] = dl.make_pdf(info)
    if key in st.session_state:
        st.download_button("Download the PDF", st.session_state[key], file_name=f"summary_{pid}.pdf",
                           mime="application/pdf", key=f"pdf_dl_{pid}")
