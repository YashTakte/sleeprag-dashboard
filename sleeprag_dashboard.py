"""
SleepRAG Results Dashboard
Run:  streamlit run sleeprag_dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(page_title="SleepRAG Results", layout="wide", initial_sidebar_state="expanded")

PRIMARY = "#0F172A"
ACCENT = "#3B82F6"
ACCENT_DARK = "#1D4ED8"
SUCCESS = "#059669"
WARNING = "#D97706"
MUTED = "#94A3B8"
LIGHT_BG = "#F8FAFC"
BORDER = "#E2E8F0"

SPEC_PALETTE = {
    "spec_primary_reference": "#64748B",
    "spec_obesity_binary": "#3B82F6",
    "spec_comorbidity_alt": "#F59E0B",
    "spec_med_classes": "#059669",
}
SPEC_LABELS = {
    "spec_primary_reference": "Primary reference",
    "spec_obesity_binary": "Obesity binary",
    "spec_comorbidity_alt": "Comorbidity alt",
    "spec_med_classes": "Medication classes",
}
OUTCOME_LABELS = {
    "sleep_efficiency": "Sleep Efficiency",
    "ahi_events_per_hour": "AHI (Apnea-Hypopnea Index)",
    "oahi_events_per_hour": "OAHI (Obstructive AHI)",
    "odi_events_per_hour": "ODI (Oxygen Desaturation Index)",
}
SPEC_ORDER = ["spec_primary_reference", "spec_obesity_binary", "spec_comorbidity_alt", "spec_med_classes"]

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}

    /* Sidebar background */
    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, #1E3A5F 0%, #0F172A 100%);
        padding-top: 1.5rem;
    }}
    [data-testid="stSidebar"] hr {{ border-color: rgba(255,255,255,0.08); margin: 1rem 0; }}

    /* Sidebar text: target labels and paragraphs, NOT the dropdown value */
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stMarkdown,
    [data-testid="stSidebar"] small {{
        color: #CBD5E1 !important;
    }}

    /* Dropdown labels above the box */
    [data-testid="stSidebar"] .stSelectbox label {{
        font-weight: 700 !important;
        font-size: 1.05rem !important;
        color: white !important;
    }}

    /* Dropdown selected value INSIDE the white box: dark text */
    [data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] {{
        color: {PRIMARY} !important;
    }}
    [data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] span {{
        color: {PRIMARY} !important;
        font-size: 0.92rem !important;
        font-weight: 500 !important;
    }}
    [data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] div {{
        color: {PRIMARY} !important;
    }}
    [data-testid="stSidebar"] .stSelectbox svg {{
        fill: {MUTED} !important;
    }}
    /* Dropdown menu items */
    [data-testid="stSidebar"] ul[role="listbox"] li {{
        color: {PRIMARY} !important;
    }}

    .sb-title {{
        font-size: 1.7rem;
        font-weight: 800;
        color: white !important;
        letter-spacing: -0.3px;
        margin-bottom: 0.6rem;
    }}
    .sb-motiv {{
        font-size: 0.95rem;
        line-height: 1.65;
        color: #94A3B8 !important;
        margin-bottom: 0.5rem;
    }}

    .stApp {{ background: {LIGHT_BG}; }}

    .page-title {{ font-size: 1.85rem; font-weight: 800; color: {PRIMARY}; letter-spacing: -0.4px; margin-bottom: 0.15rem; }}
    .page-sub {{ font-size: 0.92rem; color: {MUTED}; margin-bottom: 1.8rem; line-height: 1.55; }}
    .sec-head {{
        font-size: 1.1rem; font-weight: 700; color: {PRIMARY};
        margin: 2rem 0 0.25rem 0; padding-bottom: 0.3rem;
        border-bottom: 2.5px solid {ACCENT}; display: inline-block;
    }}
    .sec-desc {{ font-size: 0.85rem; color: #64748B; margin-bottom: 1rem; line-height: 1.6; }}
    .note-box {{
        background: white; border-left: 4px solid {ACCENT}; border-radius: 0 8px 8px 0;
        padding: 0.85rem 1.1rem; margin: 0.7rem 0 1rem 0;
        font-size: 0.85rem; color: #334155; line-height: 1.55;
        box-shadow: 0 1px 2px rgba(0,0,0,0.04);
    }}
    .note-box b {{ color: {PRIMARY}; }}

    .kpi {{
        background: white; border: 1px solid {BORDER}; border-radius: 10px;
        padding: 1rem 0.7rem; text-align: center;
        box-shadow: 0 1px 2px rgba(0,0,0,0.04);
    }}
    .kv {{ font-size: 1.55rem; font-weight: 800; line-height: 1.2; }}
    .kv.g {{ color: {SUCCESS}; }}
    .kv.b {{ color: {ACCENT_DARK}; }}
    .kv.d {{ color: {PRIMARY}; }}
    .kl {{ font-size: 0.68rem; color: {MUTED}; margin-top: 0.3rem; text-transform: uppercase; letter-spacing: 0.5px; font-weight: 600; }}

    #MainMenu {{visibility:hidden;}}
    footer {{visibility:hidden;}}
    .stDeployButton {{display:none;}}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
PSM_DIR = Path(__file__).parent


@st.cache_data
def load_effects():
    try:
        return pd.read_csv(PSM_DIR / "psm_model_selection_effects_all_specs.csv")
    except Exception:
        return pd.DataFrame()


@st.cache_data
def load_balance():
    try:
        return pd.read_csv(PSM_DIR / "psm_model_selection_balance_all_specs.csv")
    except Exception:
        return pd.DataFrame()


def is_sig(row):
    return row["ci95_low"] > 0 or row["ci95_high"] < 0


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="sb-title">SleepRAG</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sb-motiv">'
        "Does adenotonsillectomy actually improve sleep in children? "
        "This dashboard presents causal estimates from propensity score "
        "matching on the NCH Sleep DataBank, guided by a literature-derived "
        "causal graph built from 2,000+ PubMed abstracts."
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    effects_df = load_effects()
    balance_df = load_balance()

    if effects_df.empty:
        st.error("Effects CSV not found.")
        st.stop()

    macro = effects_df[effects_df["n_pairs"] > 0]["outcome"].unique()
    outcome_opts = [o for o in OUTCOME_LABELS if o in macro]

    selected_outcome = st.selectbox("Outcome", outcome_opts, format_func=lambda x: OUTCOME_LABELS[x])
    selected_spec = st.selectbox("Specification", SPEC_ORDER, format_func=lambda x: SPEC_LABELS[x], index=SPEC_ORDER.index("spec_med_classes"))

    st.markdown("---")
    st.markdown(
        '<div class="sb-motiv">'
        "Use the filters above to explore how the "
        "treatment effect changes across different "
        "outcomes and covariate encodings."
        "</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Header + KPIs
# ---------------------------------------------------------------------------
st.markdown('<div class="page-title">Propensity Score Matching Results</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="page-sub">'
    "Four PSM specifications tested whether the treatment effect survives different "
    "covariate encodings. Use the sidebar to switch outcomes and specifications."
    "</div>",
    unsafe_allow_html=True,
)

sel_row = effects_df[(effects_df["spec_name"] == selected_spec) & (effects_df["outcome"] == selected_outcome)]
sel_bal = balance_df[balance_df["spec_name"] == selected_spec] if not balance_df.empty else pd.DataFrame()

c1, c2, c3, c4, c5 = st.columns(5)

if not sel_row.empty:
    r = sel_row.iloc[0]
    ate = r["ate_matched"]
    pairs = int(r["n_pairs"])
    ctrl = r["mean_control"]
    significant = is_sig(r)

    ate_str = f"+{ate:.2f}" if ate > 0 else f"{ate:.2f}"
    c1.markdown(f'<div class="kpi"><div class="kv {"g" if significant else "d"}">{ate_str}</div><div class="kl">Treatment effect (ATE)</div></div>', unsafe_allow_html=True)

    if ctrl != 0:
        rel = abs(ate / ctrl) * 100
        c2.markdown(f'<div class="kpi"><div class="kv b">{rel:.1f}%</div><div class="kl">Relative change</div></div>', unsafe_allow_html=True)

    c3.markdown(f'<div class="kpi"><div class="kv d">{pairs}</div><div class="kl">Matched pairs</div></div>', unsafe_allow_html=True)

    sig_class = "g" if significant else "d"
    c4.markdown(f'<div class="kpi"><div class="kv {sig_class}">{"Yes" if significant else "No"}</div><div class="kl">Significant (95% CI)</div></div>', unsafe_allow_html=True)

    if not sel_bal.empty:
        mx = sel_bal["abs_smd_after"].max()
        c5.markdown(f'<div class="kpi"><div class="kv {"g" if mx < 0.10 else "d"}">{mx:.4f}</div><div class="kl">Max |SMD| after match</div></div>', unsafe_allow_html=True)

# Brief KPI explainer
st.markdown(
    '<div class="note-box">'
    "<b>ATE</b> (Average Treatment Effect) is the difference in means between matched treated and control groups. "
    "<b>Relative change</b> is the ATE as a percentage of the control group mean, showing how large the effect is "
    "compared to the untreated baseline. "
    "<b>Max |SMD|</b> is the worst remaining covariate imbalance after matching (below 0.10 is considered good)."
    "</div>",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Forest plot
# ---------------------------------------------------------------------------
st.markdown(f'<div class="sec-head">Forest plot: {OUTCOME_LABELS[selected_outcome]}</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sec-desc">'
    "Each dot is the estimated treatment effect for one specification. The horizontal line "
    "is the 95% confidence interval. If it does not cross the dotted zero line, the result "
    "is statistically significant. The highlighted marker is the specification selected in the sidebar."
    "</div>",
    unsafe_allow_html=True,
)

pos_good = selected_outcome == "sleep_efficiency"
st.markdown(
    f'<div class="note-box">{"A <b>positive</b> ATE means surgery improved sleep efficiency." if pos_good else "A <b>negative</b> ATE means fewer events per hour (improvement)."}</div>',
    unsafe_allow_html=True,
)

odf = effects_df[(effects_df["outcome"] == selected_outcome) & (effects_df["n_pairs"] > 0)]
odf = odf.set_index("spec_name").reindex(SPEC_ORDER).dropna(subset=["ate_matched"]).reset_index()

fig = go.Figure()
for i, (_, row) in enumerate(odf.iterrows()):
    s = row["spec_name"]
    a, lo, hi = row["ate_matched"], row["ci95_low"], row["ci95_high"]
    is_sel = s == selected_spec
    c = SPEC_PALETTE.get(s, MUTED)
    lab = SPEC_LABELS.get(s, s)
    opacity = 1.0 if is_sel else 0.45

    fig.add_trace(go.Scatter(x=[lo, hi], y=[i, i], mode="lines", line=dict(width=5 if is_sel else 2.5, color=c), opacity=opacity, showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=[a], y=[i], mode="markers",
        marker=dict(size=18 if is_sel else 10, color=c, symbol="diamond" if is_sel else "circle", line=dict(width=2.5, color="white")),
        opacity=opacity, name=lab,
        hovertemplate=f"<b>{lab}</b><br>ATE: {a:.3f}<br>CI: [{lo:.3f}, {hi:.3f}]<extra></extra>",
    ))

fig.add_vline(x=0, line_dash="dot", line_color=MUTED, line_width=1.5)
fig.update_layout(
    yaxis=dict(tickvals=list(range(len(odf))), ticktext=[f"<b>{SPEC_LABELS[s]}</b>" if s == selected_spec else SPEC_LABELS[s] for s in odf["spec_name"]], autorange="reversed", tickfont=dict(size=12)),
    xaxis=dict(title=dict(text="Average Treatment Effect (ATE)", font=dict(size=11, color=MUTED)), gridcolor=BORDER, zerolinecolor=MUTED),
    height=280, margin=dict(l=220, r=40, t=12, b=50),
    plot_bgcolor="white", paper_bgcolor=LIGHT_BG,
    legend=dict(orientation="h", yanchor="bottom", y=-0.45, font=dict(size=11)),
    font=dict(family="Inter, sans-serif"),
)
fig.update_yaxes(gridcolor="white")
st.plotly_chart(fig, width="stretch")

# ---------------------------------------------------------------------------
# Results table
# ---------------------------------------------------------------------------
st.markdown(f'<div class="sec-head">Detailed results: {OUTCOME_LABELS[selected_outcome]}</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sec-desc">'
    "<b>ATE</b> = difference in outcome means between matched groups. "
    "<b>CI</b> = 95% confidence interval. "
    "<b>|SMD|</b> = covariate balance after matching (below 0.10 is good)."
    "</div>",
    unsafe_allow_html=True,
)

tdf = odf.copy()
tdf["Specification"] = tdf["spec_name"].map(SPEC_LABELS)
tdf["Significant"] = tdf.apply(lambda r: "Yes" if is_sig(r) else "No", axis=1)
if not balance_df.empty:
    ba = balance_df.groupby("spec_name").agg(mean_smd=("abs_smd_after", "mean"), max_smd=("abs_smd_after", "max")).reset_index()
    tdf = tdf.merge(ba, on="spec_name", how="left")

cols = ["Specification", "n_pairs", "mean_treated", "mean_control", "ate_matched", "ci95_low", "ci95_high", "Significant"]
if "mean_smd" in tdf.columns:
    cols += ["mean_smd", "max_smd"]
rn = {"n_pairs": "Pairs", "mean_treated": "Treated", "mean_control": "Control", "ate_matched": "ATE", "ci95_low": "CI low", "ci95_high": "CI high", "mean_smd": "Mean |SMD|", "max_smd": "Max |SMD|"}
disp = tdf[[c for c in cols if c in tdf.columns]].rename(columns=rn)
for c in disp.select_dtypes(include=[np.number]).columns:
    disp[c] = disp[c].round(3)
st.dataframe(disp, width="stretch", hide_index=True)

# ---------------------------------------------------------------------------
# Balance loveplot
# ---------------------------------------------------------------------------
if not balance_df.empty:
    st.markdown(f'<div class="sec-head">Covariate balance: {SPEC_LABELS[selected_spec]}</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sec-desc">'
        "<b>Hollow gray dots</b> = imbalance before matching. "
        "<b>Blue dots</b> = imbalance after. "
        "The amber line at 0.10 is the standard threshold. "
        "Every blue dot to its left means that covariate is well-balanced."
        "</div>",
        unsafe_allow_html=True,
    )

    sb = balance_df[balance_df["spec_name"] == selected_spec].sort_values("abs_smd_after", ascending=True)

    if not sb.empty:
        fl = go.Figure()
        for _, row in sb.iterrows():
            fl.add_trace(go.Scatter(x=[row["abs_smd_before"], row["abs_smd_after"]], y=[row["covariate"]]*2, mode="lines", line=dict(width=1.2, color="#CBD5E1"), showlegend=False, hoverinfo="skip"))
        fl.add_trace(go.Scatter(x=sb["abs_smd_before"], y=sb["covariate"], mode="markers", name="Before", marker=dict(size=9, color=MUTED, symbol="circle-open", line=dict(width=2.5, color=MUTED)), hovertemplate="%{y}: %{x:.4f}<extra>Before</extra>"))
        fl.add_trace(go.Scatter(x=sb["abs_smd_after"], y=sb["covariate"], mode="markers", name="After", marker=dict(size=9, color=ACCENT), hovertemplate="%{y}: %{x:.4f}<extra>After</extra>"))
        fl.add_vline(x=0.1, line_dash="dash", line_color=WARNING, line_width=1.5, annotation_text="0.10", annotation_position="top right", annotation_font=dict(size=10, color=WARNING))
        fl.update_layout(
            xaxis=dict(title=dict(text="Absolute |SMD|", font=dict(size=11, color=MUTED)), gridcolor=BORDER, range=[0, max(sb["abs_smd_before"].max()*1.15, 0.2)]),
            yaxis=dict(gridcolor="white", tickfont=dict(size=11)),
            height=max(380, len(sb)*28), margin=dict(l=240, r=40, t=12, b=50),
            plot_bgcolor="white", paper_bgcolor=LIGHT_BG,
            legend=dict(orientation="h", yanchor="bottom", y=-0.12, font=dict(size=11)),
            font=dict(family="Inter, sans-serif"),
        )
        st.plotly_chart(fl, width="stretch")

        ma = sb["abs_smd_after"].mean()
        mx = sb["abs_smd_after"].max()
        mb = sb["abs_smd_before"].mean()
        mxb = sb["abs_smd_before"].max()
        w = sb.loc[sb["abs_smd_after"].idxmax(), "covariate"]

        bc = st.columns(3)
        bc[0].metric("Mean |SMD| after", f"{ma:.4f}", delta=f"{ma-mb:.4f} from before", delta_color="inverse")
        bc[1].metric("Max |SMD| after", f"{mx:.4f}", delta=f"{mx-mxb:.4f} from before", delta_color="inverse")
        bc[2].metric("Worst covariate", w)

        if mx < 0.10:
            v = "All covariates are below the 0.10 threshold. The matched groups are well-balanced."
        elif mx < 0.15:
            v = f"Most covariates are balanced. <b>{w}</b> is slightly above threshold at {mx:.4f}."
        else:
            v = f"<b>{w}</b> has residual imbalance at {mx:.4f}, which may affect the estimate."
        st.markdown(f'<div class="note-box">{v}</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Significance heatmap
# ---------------------------------------------------------------------------
st.markdown('<div class="sec-head">Significance overview</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sec-desc">'
    "Green = significant (95% CI excludes zero). Gray = not significant. "
    "A full green row means the finding is robust across all covariate encodings."
    "</div>",
    unsafe_allow_html=True,
)

mall = effects_df[effects_df["n_pairs"] > 0].copy()
if not mall.empty:
    mall["s"] = mall.apply(is_sig, axis=1)
    mall["Spec"] = mall["spec_name"].map(SPEC_LABELS)
    mall["Out"] = mall["outcome"].map(OUTCOME_LABELS)
    mall = mall.dropna(subset=["Out"])
    pv = mall.pivot_table(index="Out", columns="Spec", values="s", aggfunc="first")
    co = [SPEC_LABELS[s] for s in SPEC_ORDER if SPEC_LABELS[s] in pv.columns]
    ro = [OUTCOME_LABELS[o] for o in OUTCOME_LABELS if OUTCOME_LABELS[o] in pv.index]
    pv = pv.reindex(index=ro, columns=co)
    z = pv.fillna(False).astype(int).values
    tx = np.where(z == 1, "Significant", "Not significant")

    fh = go.Figure(data=go.Heatmap(
        z=z, x=pv.columns.tolist(), y=pv.index.tolist(),
        text=tx, texttemplate="%{text}", textfont=dict(size=12, color="white"),
        colorscale=[[0, "#CBD5E1"], [1, SUCCESS]], showscale=False, hoverinfo="skip", xgap=4, ygap=4,
    ))
    fh.update_layout(
        height=230, margin=dict(l=240, r=40, t=10, b=10),
        plot_bgcolor=LIGHT_BG, paper_bgcolor=LIGHT_BG,
        font=dict(family="Inter, sans-serif"),
        xaxis=dict(tickfont=dict(size=11), side="top"),
        yaxis=dict(tickfont=dict(size=12), autorange="reversed"),
    )
    st.plotly_chart(fh, width="stretch")

    st.markdown(
        '<div class="note-box">'
        "<b>Sleep Efficiency</b> and <b>ODI</b> are significant in every specification. "
        "<b>AHI</b> and <b>OAHI</b> reach significance only when obesity is encoded as a "
        "binary flag, indicating sensitivity to that encoding choice."
        "</div>",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Micro outcomes
# ---------------------------------------------------------------------------
mi = effects_df[effects_df["n_pairs"] == 0]
if not mi.empty:
    st.markdown('<div class="sec-head">Micro outcomes</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sec-desc">'
        "Brain-level sleep measures (deep sleep power, spindle density, fragmentation) "
        "test whether surgery restores sleep architecture, not just airway patency."
        "</div>",
        unsafe_allow_html=True,
    )
    names = sorted(mi["outcome"].str.replace("_", " ").str.title().unique())
    st.markdown(
        f'<div class="note-box">'
        f"<b>Zero matched pairs</b> with valid values for: {', '.join(names)}. "
        f"Features were extracted from raw EDFs but no treated-control pair both had "
        f"non-missing micro values after matching. Reported as exploratory."
        f"</div>",
        unsafe_allow_html=True,
    )
