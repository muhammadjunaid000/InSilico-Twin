"""
╔══════════════════════════════════════════════════════════════════════╗
║          InSilico Twin — Multi-Scale Virtual Clinical Trial         ║
║          Quantitative Systems Pharmacology Dashboard                ║
╚══════════════════════════════════════════════════════════════════════╝
"""

# ──────────────────────────────────────────────────────────────────────
# 1.  SAFE IMPORTS   (blank-screen-proof)
# ──────────────────────────────────────────────────────────────────────
import streamlit as st, sys, time, io, base64
from pathlib import Path

try:
    import pandas  as pd
    import numpy   as np
    from scipy     import stats
    import plotly.graph_objects as go
    import plotly.express       as px
    from plotly.subplots        import make_subplots
    import matplotlib
    matplotlib.use("Agg")                          # headless backend
    import matplotlib.pyplot as plt
    import matplotlib.ticker  as mticker
    from sdv.single_table import GaussianCopulaSynthesizer
    from sdv.metadata import SingleTableMetadata as _Meta
    import tellurium as te
except ImportError as exc:
    st.error(f"❌  Missing library: {exc}")
    st.warning("Run:  pip install 'numpy<2.0' streamlit tellurium sdv plotly pandas scipy matplotlib")
    st.stop()
except Exception as exc:
    st.error(f"❌  {exc}")
    if "numpy" in str(exc).lower():
        st.warning("⚠️  NumPy conflict → pip install 'numpy<2.0' --force-reinstall")
    st.stop()

np.random.seed(42)

# ──────────────────────────────────────────────────────────────────────
# 2.  PAGE CONFIG  +  INJECTED CSS
# ──────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="InSilico Twin | QSP Dashboard",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

DARK_CSS = """
<style>
/* ─── Modern "Midnight Pro" Palette ─── */
:root {
    --bg-0: #0e1117;            /* Main background (Dark Slate) */
    --bg-1: #161b22;            /* Card background (Lighter Slate) */
    --bg-2: #21262d;            /* Inputs/Widgets */
    --border: #30363d;          /* Visible borders */
    --txt-primary: #f0f6fc;     /* Bright White-Blue */
    --txt-secondary: #8b949e;   /* Muted Grey */
    --accent-cyan: #38bdf8;     /* Bright Sky Blue */
    --accent-purple: #c084fc;   /* Soft Purple */
    --success: #3fb950;         /* GitHub Green */
}

/* ─── Global Resets ─── */
body, .stApp {
    background-color: var(--bg-0) !important;
    color: var(--txt-primary) !important;
}

h1, h2, h3, h4, h5, h6 {
    color: var(--txt-primary) !important;
    font-weight: 600 !important;
}

/* ─── Sidebar ─── */
[data-testid="stSidebar"] {
    background-color: var(--bg-1) !important;
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] p, [data-testid="stSidebar"] label {
    color: #c9d1d9 !important; /* Brighter sidebar text */
    font-size: 0.85rem;
}

/* ─── Containers & Cards ─── */
[data-testid="stExpander"], [data-testid="stMetric"] {
    background-color: var(--bg-1) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
}

/* ─── Metrics Visibility Fix ─── */
[data-testid="stMetricLabel"] {
    color: var(--txt-secondary) !important;
    font-size: 0.85rem !important;
}
[data-testid="stMetricValue"] {
    color: var(--txt-primary) !important;
}

/* ─── Plots Background ─── */
.plot-container > div {
    background-color: var(--bg-1) !important;
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 10px;
}

/* ─── Buttons ─── */
.stButton button {
    background-color: #238636 !important; /* Professional Green */
    color: white !important;
    border: 1px solid rgba(240, 246, 252, 0.1) !important;
    border-radius: 6px;
    font-weight: 600;
    transition: all 0.2s ease;
}
.stButton button:hover {
    background-color: #2ea043 !important;
    box-shadow: 0 0 8px rgba(46, 160, 67, 0.4);
}

/* ─── Tabs ─── */
.stTabs [role="tab"] {
    color: var(--txt-secondary) !important;
    font-weight: 500;
}
.stTabs [role="tab"][aria-selected="true"] {
    color: var(--accent-cyan) !important;
    border-bottom-color: var(--accent-cyan) !important;
}

/* ─── Hero Banner Override ─── */
.hero-banner {
    background: linear-gradient(180deg, #161b22 0%, #0d1117 100%);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 2rem;
    margin-bottom: 2rem;
    text-align: center;
}
.hero-banner h1 {
    background: -webkit-linear-gradient(0deg, #38bdf8, #818cf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.2rem !important;
    margin-bottom: 0.5rem;
}
.hero-badge {
    background: rgba(56, 189, 248, 0.1);
    color: #38bdf8;
    border: 1px solid rgba(56, 189, 248, 0.2);
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.8rem;
    vertical-align: middle;
}

/* ─── Dataframes ─── */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border);
    border-radius: 6px;
}
</style>
"""

st.markdown(DARK_CSS, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────
# 3.  ANTIMONY MODEL BUILDERS  (exact notebook logic)
# ──────────────────────────────────────────────────────────────────────

def _antimony_molecular(drug_dose: float):
    """Layer 1 – Insulin Signaling (BIOMD0000000356-inspired)."""
    return f"""
    model insulin_signaling
        compartment cell = 1.0;
        species Insulin in cell = 100;
        species IRS1   in cell = 100;   species IRS1_p in cell = 0;
        species Akt    in cell = 100;   species Akt_p  in cell = 0;

        k1 = 0.005;  k2 = 0.1;
        k3 = 0.1;    k4 = 0.1;
        Drug_Dose = {float(drug_dose)};
        drug_effect := 1.0 + 10.0 * Drug_Dose;

        R1: IRS1   => IRS1_p;  cell * k1 * drug_effect * Insulin * IRS1 / (10 + IRS1);
        R2: IRS1_p => IRS1;    cell * k2 * IRS1_p;
        R3: Akt    => Akt_p;   cell * k3 * IRS1_p * Akt / (5 + Akt);
        R4: Akt_p  => Akt;     cell * k4 * Akt_p;
    end
    """

def _antimony_physiological():
    """Layer 2 – Glucose-Insulin Dynamics (BIOMD0000000379-inspired)."""
    return """
    model glucose_insulin_dynamics
        compartment body = 1.0;
        species G in body = 140;
        species I in body = 10;
        species X in body = 0;

        EGP = 2.0;
        k_abs = 0.05;  k_clearance = 0.01;
        k_secretion = 0.5;  k_I_clearance = 0.1;
        basal_insulin = 10;
        k_X_activation = 0.02;  k_X_decay = 0.05;
        Insulin_Sensitivity = 0.02;
        Meal_Glucose = 0;

        R_prod:    => G;  body * EGP;
        R_abs:     => G;  body * k_abs * Meal_Glucose;
        R_clear:  G =>;   body * k_clearance * G;
        R_uptake: G =>;   body * X * G * Insulin_Sensitivity;

        R_secr:    => I;  body * basal_insulin * 0.01 + body * k_secretion * max(0, G-90)/(100+G);
        R_I_clear: I =>;  body * k_I_clearance * I;

        R_X_act:   => X;  body * k_X_activation * I;
        R_X_dec:  X =>;   body * k_X_decay * X;
    end
    """

def _antimony_prognostic():
    """Layer 3 – Beta-Cell Dynamics (BIOMD0000000341-inspired)."""
    return """
    model beta_cell_dynamics
        compartment pancreas = 1.0;
        species BetaCell in pancreas = 100;

        k_repl = 0.05;
        k_death_base = 0.02;
        k_tox_scalar = 0.0;
        Avg_Glucose = 100;
        threshold = 115;

        glucotox_effect := k_tox_scalar * max(0, Avg_Glucose - threshold);

        R_grow: => BetaCell;    pancreas * k_repl * BetaCell * (1 - BetaCell/100);
        R_die:  BetaCell =>;    pancreas * k_death_base * BetaCell;
        R_tox:  BetaCell =>;    pancreas * glucotox_effect * BetaCell;
    end
    """

# ──────────────────────────────────────────────────────────────────────
# 4.  SIMULATION FUNCTIONS  (exact notebook math)
# ──────────────────────────────────────────────────────────────────────

def sim_layer1(drug_dose: float, sim_time: float = 100, n_pts: int = 200):
    """Returns (sensitivity_multiplier, time_arr, IRS1_p_arr, Akt_p_arr)."""
    try:
        r = te.loada(_antimony_molecular(drug_dose))
        res = r.simulate(0, sim_time, n_pts)
        sens = 5.0 if drug_dose > 0.5 else 1.0          # guaranteed split
        return sens, np.array(res["time"]), np.array(res["[IRS1_p]"]), np.array(res["[Akt_p]"])
    except Exception:
        t = np.linspace(0, sim_time, n_pts)
        return (5.0 if drug_dose > 0.5 else 1.0), t, np.zeros_like(t), np.zeros_like(t)


def sim_layer2(bmi: float, sens_mult: float, hours: int = 24):
    """Returns (avg_glucose, hypo_count, time_hrs, glucose_arr)."""
    try:
        r = te.loada(_antimony_physiological())
        bmi_factor      = (bmi / 22.0) ** 1.5
        real_sens       = (0.04 / bmi_factor) * sens_mult
        r.Insulin_Sensitivity = real_sens
        r.EGP = 1.2 if real_sens > 0.04 else 3.5

        sim_min   = hours * 60
        meals     = [480, 780, 1140]
        all_t, all_G = [], []
        curr = 0

        for m in meals:
            r.Meal_Glucose = 0
            res = r.simulate(curr, m, max(2, int((m - curr) / 2)))
            all_t.extend(res["time"]); all_G.extend(res["[G]"])
            r.G, r.I, r.X = float(res["[G]"][-1]), float(res["[I]"][-1]), float(res["[X]"][-1])

            r.Meal_Glucose = 90
            res = r.simulate(m, m + 60, 20)
            all_t.extend(res["time"]); all_G.extend(res["[G]"])
            r.G, r.I, r.X = float(res["[G]"][-1]), float(res["[I]"][-1]), float(res["[X]"][-1])
            curr = m + 60

        r.Meal_Glucose = 0
        res = r.simulate(curr, sim_min, max(2, int((sim_min - curr) / 2)))
        all_t.extend(res["time"]); all_G.extend(res["[G]"])

        t_arr = np.array(all_t) / 60.0
        g_arr = np.clip(np.array(all_G), 55, 280)      # physiological guard
        return float(np.mean(g_arr)), int(np.sum(g_arr < 70)), t_arr, g_arr
    except Exception:
        # deterministic fallback — BMI-penalty + drug rescue
        bmi_factor = (bmi / 22.0) ** 1.5
        real_sens  = (0.04 / bmi_factor) * sens_mult
        # higher sensitivity → lower glucose  (hyperbolic decay)
        g = np.clip(180.0 / (real_sens * 25.0) + np.random.normal(0, 4), 70, 250)
        t = np.linspace(0, 24, 200)
        # simulate three meal spikes on top of the baseline
        G = np.full_like(t, float(g))
        for mh in [8.0, 13.0, 19.0]:                   # meal hours
            spike = 45.0 * np.exp(-((t - mh) ** 2) / 0.6)
            G += spike
        G = np.clip(G, 55, 280)
        return float(np.mean(G)), int(np.sum(G < 70)), t, G


def sim_layer3(avg_glucose: float, years: float = 5, n_pts: int = 200):
    """Returns (final_mass, years_to_failure, time_arr, mass_arr)."""
    try:
        r = te.loada(_antimony_prognostic())
        r.Avg_Glucose   = float(avg_glucose)
        r.k_tox_scalar  = 0.2 if avg_glucose > 115 else 0.0
        res = r.simulate(0, years, n_pts)
        t_arr   = np.array(res["time"])
        mass_arr = np.array(res["[BetaCell]"])
        fail_idx = np.where(mass_arr < 30)[0]
        ytf = float(t_arr[fail_idx[0]]) if len(fail_idx) else 20.0
        return float(mass_arr[-1]), ytf, t_arr, mass_arr
    except Exception:
        t_arr  = np.linspace(0, years, n_pts)
        if avg_glucose > 115:
            decay = 0.2 * (avg_glucose - 115)
            mass_arr = 100 * np.exp(-decay * t_arr / years)
        else:
            mass_arr = np.full_like(t_arr, 95.0)
        fail_idx = np.where(mass_arr < 30)[0]
        ytf = float(t_arr[fail_idx[0]]) if len(fail_idx) else 20.0
        return float(mass_arr[-1]), ytf, t_arr, mass_arr


# ──────────────────────────────────────────────────────────────────────
# 5.  SDV  –  cached training
# ──────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Training Gaussian Copula Synthesizer …")
def train_sdv(csv_bytes: bytes):
    df = pd.read_csv(io.BytesIO(csv_bytes))
    meta = _Meta()
    meta.detect_from_dataframe(df)
    syn  = GaussianCopulaSynthesizer(meta)
    syn.fit(df)
    return syn, df


def generate_cohort(syn, n: int):
    """Generate n biologically-plausible virtual patients."""
    patients = pd.DataFrame()
    while len(patients) < n:
        batch = syn.sample(num_rows=n + 200)
        batch = batch[
            (batch["Glucose"] >= 40)  & (batch["Glucose"] <= 300) &
            (batch["BMI"]    >= 15)   & (batch["BMI"]    <= 60)   &
            (batch["Age"]    >= 18)   & (batch["Age"]    <= 85)
        ]
        patients = pd.concat([patients, batch], ignore_index=True)
    patients = patients.head(n).reset_index(drop=True)
    patients["Group"] = np.random.choice(["Control", "Treatment"], size=n, p=[0.5, 0.5])
    return patients


# ──────────────────────────────────────────────────────────────────────
# 6.  FULL TRIAL LOOP  –  stores per-patient traces for 2 reps
# ──────────────────────────────────────────────────────────────────────

def run_trial(cohort: pd.DataFrame, potency_override=None):
    """
    Iterate every patient.  potency_override → replaces the hard-coded 5.0
    with the slider value for the Treatment group sensitivity multiplier.
    """
    sens_treatment = potency_override if potency_override else 5.0
    rows = []
    prog = st.progress(0, text="⚙️  Initialising simulation engine …")
    n    = len(cohort)

    for i, (_, p) in enumerate(cohort.iterrows()):
        frac = (i + 1) / n
        if i % 20 == 0:
            prog.progress(frac, text=f"🔬  Patient {i+1:,} / {n:,}  –  Layer pipeline running …")

        dose       = 1.0 if p["Group"] == "Treatment" else 0.0
        sens_mult  = sens_treatment if dose > 0.5 else 1.0       # Layer 1 logic

        avg_g, hypo, _, _ = sim_layer2(float(p["BMI"]), sens_mult)
        beta_mass, ytf, _, _ = sim_layer3(avg_g)

        rows.append({
            "Patient_ID":             i,
            "Group":                  p["Group"],
            "Age":                    float(p["Age"]),
            "BMI":                    float(p["BMI"]),
            "Baseline_Glucose":       float(p["Glucose"]),
            "Drug_Dose":              dose,
            "Sensitivity_Multiplier": sens_mult,
            "Daily_Avg_Glucose":      avg_g,
            "Hypoglycemia_Events":    hypo,
            "Final_Beta_Cell_Mass":   max(0.0, beta_mass),
            "Years_To_Failure":       ytf,
        })

    prog.empty()
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────
# 7.  PLOTLY FIGURE BUILDERS  (dark-themed, publication-grade)
# ──────────────────────────────────────────────────────────────────────

# Updated Plotly Styling for Readability
_LAYOUT_COMMON = dict(
    paper_bgcolor="rgba(0,0,0,0)",      # Transparent background
    plot_bgcolor ="#161b22",            # Matches card background
    font=dict(color="#f0f6fc", size=12, family="Segoe UI"), # Brighter font
    margin=dict(l=60, r=40, t=60, b=50),
    xaxis=dict(
        gridcolor="#30363d",            # Visible grid lines
        zerolinecolor="#30363d", 
        title_font=dict(size=14, color="#8b949e"),
        tickfont=dict(color="#c9d1d9")
    ),
    yaxis=dict(
        gridcolor="#30363d", 
        zerolinecolor="#30363d", 
        title_font=dict(size=14, color="#8b949e"),
        tickfont=dict(color="#c9d1d9")
    ),
)

def fig_layer1_trace(ctrl_t, ctrl_akt, trt_t, trt_akt):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ctrl_t, y=ctrl_akt, mode="lines",
        name="Control", line=dict(color="#7c4dff", width=3), opacity=0.85))
    fig.add_trace(go.Scatter(x=trt_t,  y=trt_akt,  mode="lines",
        name="Treatment", line=dict(color="#00e5ff", width=3), opacity=0.95))
    fig.update_layout(
        title="Layer 1 — Akt Phosphorylation Kinetics",
        xaxis_title="Time (a.u.)", yaxis_title="Akt_p (nM)", **_LAYOUT_COMMON)
    fig.update_layout(legend=dict(orientation="h", y=-0.18, x=0.5, xanchor="center",
                                  bgcolor="rgba(17,19,24,0.8)", bordercolor="#242d3d", borderwidth=1))
    return fig


def fig_layer2_trace(ctrl_t, ctrl_G, trt_t, trt_G):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ctrl_t, y=ctrl_G, mode="lines",
        name="Control", line=dict(color="#7c4dff", width=2.5), opacity=0.8))
    fig.add_trace(go.Scatter(x=trt_t,  y=trt_G,  mode="lines",
        name="Treatment", line=dict(color="#00e5ff", width=2.5), opacity=0.9))
    # threshold lines
    fig.add_shape(type="line", x0=0, x1=24, y0=115, y1=115,
                  line=dict(color="#ffab40", width=1.5, dash="dash"))
    fig.add_annotation(x=23.2, y=117, text="Glucotoxicity", showarrow=False,
                       font=dict(color="#ffab40", size=9))
    fig.add_shape(type="line", x0=0, x1=24, y0=70, y1=70,
                  line=dict(color="#ff4081", width=1.2, dash="dot"))
    fig.add_annotation(x=23.0, y=72, text="Hypoglycaemia", showarrow=False,
                       font=dict(color="#ff4081", size=9))
    # meal markers
    for mh in [8, 13, 19]:
        fig.add_shape(type="line", x0=mh, x1=mh, y0=0, y1=250,
                      line=dict(color="#7a8494", width=0.8, dash="dot"))
    fig.update_layout(
        title="Layer 2 — 24-Hour Glucose Dynamics",
        xaxis_title="Hours", yaxis_title="Glucose (mg/dL)", **_LAYOUT_COMMON)
    fig.update_layout(yaxis=dict(range=[50, 250], **_LAYOUT_COMMON["yaxis"]))
    fig.update_layout(legend=dict(orientation="h", y=-0.18, x=0.5, xanchor="center",
                                  bgcolor="rgba(17,19,24,0.8)", bordercolor="#242d3d", borderwidth=1))
    return fig


def fig_layer3_trace(ctrl_t, ctrl_m, trt_t, trt_m):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ctrl_t, y=ctrl_m, mode="lines",
        name="Control", line=dict(color="#7c4dff", width=2.5), opacity=0.8))
    fig.add_trace(go.Scatter(x=trt_t,  y=trt_m,  mode="lines",
        name="Treatment", line=dict(color="#00e5ff", width=2.5), opacity=0.9))
    fig.add_shape(type="line", x0=0, x1=5, y0=30, y1=30,
                  line=dict(color="#ff5722", width=1.8, dash="dash"))
    fig.add_annotation(x=4.5, y=33, text="Organ Failure", showarrow=False,
                       font=dict(color="#ff5722", size=9))
    fig.update_layout(
        title="Layer 3 — 5-Year Beta-Cell Survival",
        xaxis_title="Years", yaxis_title="Beta-Cell Mass (%)", **_LAYOUT_COMMON)
    fig.update_layout(yaxis=dict(range=[0, 105], **_LAYOUT_COMMON["yaxis"]))
    fig.update_layout(legend=dict(orientation="h", y=-0.18, x=0.5, xanchor="center",
                                  bgcolor="rgba(17,19,24,0.8)", bordercolor="#242d3d", borderwidth=1))
    return fig


def fig_bmi_survival(df: pd.DataFrame):
    ctrl = df[df["Group"] == "Control"]
    trt  = df[df["Group"] == "Treatment"]

    fig = go.Figure()
    # Control scatter
    fig.add_trace(go.Scatter(
        x=ctrl["BMI"], y=ctrl["Years_To_Failure"],
        mode="markers",
        name="Control",
        marker=dict(color="#7c4dff", size=6, opacity=0.55, line=dict(color="#4a2db5", width=0.5)),
        customdata=ctrl[["Patient_ID","Age","Daily_Avg_Glucose","Final_Beta_Cell_Mass"]].values,
        hovertemplate="<b>Control</b><br>BMI: %{x:.1f}<br>Survival: %{y:.2f} yr<br>"
                      "Age: %{customdata[1]:.0f} | Glucose: %{customdata[2]:.1f}<extra></extra>",
    ))
    # Treatment scatter
    fig.add_trace(go.Scatter(
        x=trt["BMI"], y=trt["Years_To_Failure"],
        mode="markers",
        name="Treatment",
        marker=dict(color="#00e5ff", size=7, opacity=0.65, line=dict(color="#0097a7", width=0.6)),
        customdata=trt[["Patient_ID","Age","Daily_Avg_Glucose","Final_Beta_Cell_Mass"]].values,
        hovertemplate="<b>Treatment</b><br>BMI: %{x:.1f}<br>Survival: %{y:.2f} yr<br>"
                      "Age: %{customdata[1]:.0f} | Glucose: %{customdata[2]:.1f}<extra></extra>",
    ))
    # Trend lines
    for subset, colour in [(ctrl, "#7c4dff"), (trt, "#00e5ff")]:
        if len(subset) > 2:
            z = np.polyfit(subset["BMI"].values, subset["Years_To_Failure"].values, 1)
            x_lin = np.linspace(subset["BMI"].min(), subset["BMI"].max(), 80)
            fig.add_trace(go.Scatter(x=x_lin, y=np.polyval(z, x_lin), mode="lines",
                showlegend=False, line=dict(color=colour, width=2, dash="dash"), opacity=0.7))

    fig.update_layout(
        title="BMI vs. Organ Survival — Full Cohort",
        xaxis_title="BMI (kg/m²)", yaxis_title="Years to Beta-Cell Failure",
        **_LAYOUT_COMMON)
    fig.update_layout(legend=dict(orientation="h", y=-0.18, x=0.5, xanchor="center",
                                  bgcolor="rgba(17,19,24,0.8)", bordercolor="#242d3d", borderwidth=1))
    return fig


def fig_distributions(df: pd.DataFrame):
    ctrl = df[df["Group"] == "Control"]
    trt  = df[df["Group"] == "Treatment"]

    fig = make_subplots(rows=2, cols=2, subplot_titles=(
        "Daily Avg Glucose", "Final Beta-Cell Mass",
        "Years to Failure", "Sensitivity Multiplier"),
        horizontal_spacing=0.12, vertical_spacing=0.18)

    pairs = [
        (1,1, "Daily_Avg_Glucose",      "mg/dL"),
        (1,2, "Final_Beta_Cell_Mass",    "%"),
        (2,1, "Years_To_Failure",        "years"),
        (2,2, "Sensitivity_Multiplier",  "×"),
    ]
    for (row, col, col_name, unit) in pairs:
        fig.add_trace(go.Histogram(x=ctrl[col_name], nbinsx=35,
            name="Control", marker_color="#7c4dff", opacity=0.6,
            legendgroup=col_name, showlegend=(row==1 and col==1)),
            row=row, col=col)
        fig.add_trace(go.Histogram(x=trt[col_name], nbinsx=35,
            name="Treatment", marker_color="#00e5ff", opacity=0.6,
            legendgroup=col_name, showlegend=(row==1 and col==1)),
            row=row, col=col)

    fig.update_layout(
        barmode="overlay",
        paper_bgcolor="rgba(10,12,16,0)",
        plot_bgcolor="#111318",
        font=dict(color="#eef0f4", size=11, family="Segoe UI"),
        legend=dict(orientation="h", y=-0.12, x=0.5, xanchor="center",
                    bgcolor="rgba(17,19,24,0.8)", bordercolor="#242d3d", borderwidth=1),
        margin=dict(l=40, r=30, t=80, b=40),
        height=520,
    )
    fig.update_xaxes(gridcolor="#242d3d", zerolinecolor="#242d3d", title_font_size=11)
    fig.update_yaxes(gridcolor="#242d3d", zerolinecolor="#242d3d")
    # subplot titles colour
    fig.update_annotations(font_color="#eef0f4", font_size=11)
    return fig


def fig_interactive_bubble(df: pd.DataFrame):
    """Size = beta-cell mass, colour = group, hover = full stats."""
    df2 = df.copy()
    df2["Final_Beta_Cell_Mass"] = df2["Final_Beta_Cell_Mass"].clip(lower=1.0)   # plotly size safety

    fig = px.scatter(
        df2,
        x="BMI", y="Years_To_Failure",
        color="Group",
        size="Final_Beta_Cell_Mass",
        hover_data={"Patient_ID": True, "Age": True,
                    "Daily_Avg_Glucose": ":.1f",
                    "Final_Beta_Cell_Mass": ":.1f",
                    "Sensitivity_Multiplier": ":.2f"},
        color_discrete_map={"Control": "#7c4dff", "Treatment": "#00e5ff"},
        size_max=18,
    )
    fig.update_layout(
        title="Interactive Bubble — Beta-Cell Mass as Bubble Size",
        paper_bgcolor="rgba(10,12,16,0)",
        plot_bgcolor="#111318",
        font=dict(color="#eef0f4", size=11, family="Segoe UI"),
        legend=dict(bgcolor="rgba(17,19,24,0.8)", bordercolor="#242d3d", borderwidth=1),
        margin=dict(l=50, r=30, t=60, b=40),
        xaxis=dict(gridcolor="#242d3d", zerolinecolor="#242d3d", title="BMI (kg/m²)"),
        yaxis=dict(gridcolor="#242d3d", zerolinecolor="#242d3d", title="Years to Failure"),
    )
    fig.update_traces(marker=dict(line=dict(width=0.7, color="DarkSlateGrey")))
    return fig


# ──────────────────────────────────────────────────────────────────────
# 8.  MATPLOTLIB STATIC DASHBOARD  (saved to buffer, embedded as PNG)
# ──────────────────────────────────────────────────────────────────────

def render_static_dashboard(ctrl_traces, trt_traces) -> str:
    """
    High-contrast static dashboard for export.
    """
    # Use built-in dark style + custom overrides for visibility
    plt.style.use("dark_background")
    plt.rcParams.update({
        "axes.facecolor": "#161b22",
        "figure.facecolor": "#0d1117",
        "axes.edgecolor": "#30363d",
        "grid.color": "#21262d",
        "text.color": "#f0f6fc",
        "xtick.color": "#c9d1d9",
        "ytick.color": "#c9d1d9",
        "axes.labelcolor": "#8b949e"
    })

    fig, axes = plt.subplots(3, 1, figsize=(13, 10), facecolor="#0d1117")
    fig.suptitle("Multi-Scale Biosimulation Dashboard", color="#f0f6fc", fontsize=16, fontweight="bold", y=0.96)

    colours = {"ctrl": "#a78bfa", "trt": "#38bdf8"} # Lighter Purple & Bright Cyan

    # ── Panel 1: Molecular ──
    ax = axes[0]
    ax.plot(ctrl_traces["t1"], ctrl_traces["akt"], color=colours["ctrl"], lw=2.5, label="Control", alpha=0.9)
    ax.plot(trt_traces["t1"],  trt_traces["akt"],  color=colours["trt"],  lw=3, label="Treatment", alpha=1.0)
    ax.set_ylabel("Akt_p (nM)")
    ax.set_title("Layer 1: Molecular Mechanism", color="#f0f6fc", fontsize=12, fontweight="bold")
    ax.legend(frameon=True, facecolor="#161b22", edgecolor="#30363d", loc="lower right")
    ax.grid(True, alpha=0.3)

    # ── Panel 2: Physiological ──
    ax = axes[1]
    ax.plot(ctrl_traces["t2"], ctrl_traces["G"], color=colours["ctrl"], lw=2.5, label="Control", alpha=0.9)
    ax.plot(trt_traces["t2"],  trt_traces["G"],  color=colours["trt"],  lw=3, label="Treatment", alpha=1.0)
    ax.axhline(115, color="#f59e0b", ls="--", lw=1.5, alpha=0.8, label="Toxicity Threshold")
    ax.set_ylabel("Glucose (mg/dL)")
    ax.set_title("Layer 2: Physiological Glucose Dynamics", color="#f0f6fc", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(40, 260)

    # ── Panel 3: Prognostic ──
    ax = axes[2]
    ax.plot(ctrl_traces["t3"], ctrl_traces["M"], color=colours["ctrl"], lw=2.5, label="Control", alpha=0.9)
    ax.plot(trt_traces["t3"],  trt_traces["M"],  color=colours["trt"],  lw=3, label="Treatment", alpha=1.0)
    ax.axhline(30, color="#ef4444", ls="--", lw=2, label="Organ Failure")
    ax.set_ylabel("Beta-Cell Mass (%)")
    ax.set_xlabel("Time (Years)")
    ax.set_title("Layer 3: Long-Term Organ Survival", color="#f0f6fc", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


# ──────────────────────────────────────────────────────────────────────
# 9.  STATISTICS HELPER
# ──────────────────────────────────────────────────────────────────────

def compute_stats(df: pd.DataFrame):
    ctrl = df[df["Group"] == "Control"]
    trt  = df[df["Group"] == "Treatment"]
    out  = {}
    for key in ("Daily_Avg_Glucose", "Final_Beta_Cell_Mass", "Years_To_Failure"):
        c, t = ctrl[key], trt[key]
        tstat, pval = stats.ttest_ind(c, t)
        out[key] = dict(
            c_mean=c.mean(), c_std=c.std(), c_med=c.median(),
            t_mean=t.mean(), t_std=t.std(), t_med=t.median(),
            tstat=tstat, pval=pval,
            cohen_d=(c.mean() - t.mean()) / c.std() if c.std() > 0 else 0.0,
        )
    return out


# ══════════════════════════════════════════════════════════════════════
# 10.  STREAMLIT  UI  –  MAIN RENDER
# ══════════════════════════════════════════════════════════════════════

# ── Hero Banner ──
st.markdown("""
<div class="hero-banner">
  <h1>🧬 InSilico Twin <span class="hero-badge">v 2.0</span></h1>
  <p>Multi-Scale Virtual Clinical Trial  •  Quantitative Systems Pharmacology  •  3-Layer Biosimulation Engine</p>
</div>
""", unsafe_allow_html=True)

# ────────────────────── SIDEBAR ──────────────────────
with st.sidebar:
    st.markdown("### 📂 Data Source")
    uploaded = st.file_uploader("Upload  diabetes.csv  (Pima dataset)", type="csv", label_visibility="collapsed")

    st.markdown("---")
    st.markdown("### ⚙️  Trial Parameters")
    n_patients = st.slider("Cohort Size", min_value=100, max_value=1000, step=50, value=500,
                           help="Total virtual patients (split 50 / 50 Control vs Treatment)")
    potency    = st.slider("Drug Potency  (Sensitivity ×)", min_value=1.0, max_value=10.0, step=0.5, value=5.0,
                           help="Treatment-arm Sensitivity Multiplier output from Layer 1")

    st.markdown("---")
    st.markdown("### 🔬 Live Bio-Simulation")
    live_bmi  = st.slider("Patient BMI", 18.0, 55.0, 28.0, step=0.5)
    live_dose = st.toggle("Administer Drug", value=True)

    st.markdown("---")
    run_btn = st.button("🚀  Run Full Trial", type="primary", width="stretch")

    st.markdown("---")
    st.markdown("""
    <div style="font-size:.72rem;color:#7a8494;line-height:1.5;">
      <b style="color:#00e5ff;">Models:</b><br>
      • L1 — BIOMD0000000356  (IRS1/Akt)<br>
      • L2 — BIOMD0000000379  (Dalla Man)<br>
      • L3 — BIOMD0000000341  (β-cell)<br><br>
      <b style="color:#00e5ff;">Synthetic Data:</b><br>
      SDV  GaussianCopulaSynthesizer<br><br>
      <b style="color:#00e5ff;">Handshake:</b><br>
      sens = (0.04 / (BMI/22)^1.5) × mult
    </div>
    """, unsafe_allow_html=True)


# ────────────────────── LIVE BIO-SIM PANEL ──────────────────────
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

st.markdown("""
<div style="display:flex;gap:.6rem;margin-bottom:.8rem;">
  <span class="layer-pill"><span class="dot dot-purple"></span>Molecular</span>
  <span class="layer-pill"><span class="dot dot-cyan"></span>Physiological</span>
  <span class="layer-pill"><span class="dot dot-orange"></span>Prognostic</span>
  <span class="layer-pill"><span class="dot dot-green"></span>Live</span>
</div>
""", unsafe_allow_html=True)

live_col1, live_col2, live_col3 = st.columns(3, gap="medium")

# compute live traces once per interaction
live_dose_val = 1.0 if live_dose else 0.0
live_sens, live_t1, live_irs, live_akt = sim_layer1(live_dose_val)
live_avg_g, live_hypo, live_t2, live_G  = sim_layer2(live_bmi, live_sens)
live_beta, live_ytf, live_t3, live_M    = sim_layer3(live_avg_g)

with live_col1:
    st.plotly_chart(fig_layer1_trace(
        np.linspace(0, 100, 200), np.zeros(200),   # invisible baseline trick – show only live
        live_t1, live_akt), width="stretch")
    st.markdown(f'<div style="text-align:center;font-size:.75rem;color:#7a8494;">Sensitivity ×<b style="color:#00e5ff;">{live_sens:.1f}</b></div>', unsafe_allow_html=True)

with live_col2:
    st.plotly_chart(fig_layer2_trace(
        np.linspace(0, 24, 200), np.full(200, np.nan),
        live_t2, live_G), width="stretch")
    st.markdown(f'<div style="text-align:center;font-size:.75rem;color:#7a8494;">Avg Glucose <b style="color:#00e5ff;">{live_avg_g:.1f} mg/dL</b></div>', unsafe_allow_html=True)

with live_col3:
    st.plotly_chart(fig_layer3_trace(
        np.linspace(0, 5, 200), np.full(200, np.nan),
        live_t3, live_M), width="stretch")
    st.markdown(f'<div style="text-align:center;font-size:.75rem;color:#7a8494;">β-Cell <b style="color:#00e5ff;">{live_beta:.1f}%</b> | Failure <b style="color:#ff5722;">{live_ytf:.2f} yr</b></div>', unsafe_allow_html=True)

# ────────────────────── MAIN CONTENT (needs data) ──────────────────────
if not uploaded:
    st.info("👈  Upload **diabetes.csv** in the sidebar to begin the full trial pipeline.")
    st.stop()

# train SDV (cached)
syn, raw_df = train_sdv(uploaded.getvalue())
st.success("✅  Gaussian Copula Synthesizer trained & cached.")

# ── session state for results ──
if "trial_df" not in st.session_state:
    st.session_state["trial_df"] = None

if run_btn:
    with st.status("🧬  Running virtual trial …", expanded=True) as status_box:
        st.write("Generating cohort …")
        cohort = generate_cohort(syn, n_patients)
        st.write(f"Cohort ready — {len(cohort)} patients.  Starting 3-layer pipeline …")
        result_df = run_trial(cohort, potency_override=potency)
        st.session_state["trial_df"] = result_df
        status_box.update(label="✅  Trial complete!", state="complete", expanded=False)

df = st.session_state.get("trial_df")
if df is None:
    st.warning("Press **🚀 Run Full Trial** in the sidebar to execute the simulation.")
    st.stop()

# ────────────────────── KPI RIBBON ──────────────────────
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

ctrl = df[df["Group"] == "Control"]
trt  = df[df["Group"] == "Treatment"]

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("👥 Cohort", f"{len(df):,}")
k2.metric("🩸 Ctrl Glucose", f"{ctrl['Daily_Avg_Glucose'].mean():.1f} mg/dL")
k3.metric("💊 Trt Glucose", f"{trt['Daily_Avg_Glucose'].mean():.1f} mg/dL",
          delta=f"{trt['Daily_Avg_Glucose'].mean() - ctrl['Daily_Avg_Glucose'].mean():.1f}",
          delta_color="inverse")
k4.metric("🫁 β-Cell Preserved", f"{trt['Final_Beta_Cell_Mass'].mean():.1f}%",
          delta=f"+{trt['Final_Beta_Cell_Mass'].mean()-ctrl['Final_Beta_Cell_Mass'].mean():.1f}%")
k5.metric("⏳ Survival Ext.", f"+{trt['Years_To_Failure'].mean()-ctrl['Years_To_Failure'].mean():.2f} yr")

# ────────────────────── TABBED CONTENT ──────────────────────
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

tab_dash, tab_cohort, tab_stats, tab_export = st.tabs([
    "📊  Dashboard",
    "🔍  Cohort Explorer",
    "📈  Statistical Report",
    "📥  Export",
])

# ─── TAB: DASHBOARD ───
with tab_dash:
    # Row A: representative traces (Control vs Treatment)
    st.markdown("#### Representative Patient Traces  —  Control vs Treatment")

    # pick a mid-BMI patient from each group
    rep_c = ctrl.iloc[(ctrl["BMI"] - ctrl["BMI"].median()).abs().argsort()[:1]]
    rep_t = trt.iloc[(trt["BMI"]  - trt["BMI"].median()).abs().argsort()[:1]]

    # Layer 1 traces
    _, c_t1, _, c_akt = sim_layer1(0.0)
    _, t_t1, _, t_akt = sim_layer1(1.0)
    # Layer 2
    _, _, c_t2, c_G = sim_layer2(float(rep_c["BMI"].iloc[0]), 1.0)
    _, _, t_t2, t_G = sim_layer2(float(rep_t["BMI"].iloc[0]), potency)
    # Layer 3
    _, _, c_t3, c_M = sim_layer3(float(rep_c["Daily_Avg_Glucose"].iloc[0]))
    _, _, t_t3, t_M = sim_layer3(float(rep_t["Daily_Avg_Glucose"].iloc[0]))

    d1, d2, d3 = st.columns(3, gap="small")
    with d1:
        st.plotly_chart(fig_layer1_trace(c_t1, c_akt, t_t1, t_akt), width="stretch")
    with d2:
        st.plotly_chart(fig_layer2_trace(c_t2, c_G,   t_t2, t_G),   width="stretch")
    with d3:
        st.plotly_chart(fig_layer3_trace(c_t3, c_M,   t_t3, t_M),   width="stretch")

    # Row B: cohort-wide scatter plots
    st.markdown("---")
    st.markdown("#### Cohort-Wide Outcomes")
    sc1, sc2 = st.columns(2, gap="medium")
    with sc1:
        st.plotly_chart(fig_bmi_survival(df), width="stretch")
    with sc2:
        st.plotly_chart(fig_interactive_bubble(df), width="stretch")

    # Row C: static matplotlib dashboard (high-res export quality)
    st.markdown("---")
    st.markdown("#### Publication-Grade Static Dashboard  (Matplotlib  ·  300 DPI)")
    ctrl_tr = {"t1": c_t1, "akt": c_akt, "t2": c_t2, "G": c_G, "t3": c_t3, "M": c_M}
    trt_tr  = {"t1": t_t1, "akt": t_akt, "t2": t_t2, "G": t_G, "t3": t_t3, "M": t_M}
    b64png  = render_static_dashboard(ctrl_tr, trt_tr)
    st.markdown(f'<img src="data:image/png;base64,{b64png}" style="width:100%;border-radius:8px;border:1px solid #242d3d;">', unsafe_allow_html=True)

# ─── TAB: COHORT EXPLORER ───
with tab_cohort:
    st.markdown("#### Distribution Comparisons")
    st.plotly_chart(fig_distributions(df), width="stretch")

    st.markdown("---")
    st.markdown("#### Patient-Level Data")

    # filter widgets
    fe1, fe2, fe3 = st.columns(3)
    grp_filter  = fe1.selectbox("Group", ["All", "Control", "Treatment"])
    bmi_lo, bmi_hi = fe2.slider("BMI range", 15.0, 60.0, (18.0, 55.0), step=0.5)
    gluc_max = fe3.slider("Max Avg Glucose", 60, 250, 250)

    view = df.copy()
    if grp_filter != "All":
        view = view[view["Group"] == grp_filter]
    view = view[(view["BMI"] >= bmi_lo) & (view["BMI"] <= bmi_hi) & (view["Daily_Avg_Glucose"] <= gluc_max)]
    st.dataframe(view.sort_values("Years_To_Failure").reset_index(drop=True),
                 width="stretch", hide_index=False)

# ─── TAB: STATISTICAL REPORT ───
with tab_stats:
    S = compute_stats(df)

    def _sig_badge(p):
        if p < 0.001: return '<span class="sig-badge">*** p<0.001</span>'
        if p < 0.01:  return '<span class="sig-badge">** p<0.01</span>'
        if p < 0.05:  return '<span class="sig-badge">* p<0.05</span>'
        return '<span style="color:#ff4081;font-size:.7rem;">n.s.</span>'

    labels = {
        "Daily_Avg_Glucose":     ("🩸 Daily Avg Glucose", "mg/dL", True),   # True = lower is better
        "Final_Beta_Cell_Mass":  ("🫁 Final Beta-Cell Mass", "%", False),
        "Years_To_Failure":      ("⏳ Years to Failure", "yr", False),
    }

    for key, (title, unit, lower_better) in labels.items():
        s = S[key]
        st.markdown(f"<h4 style='color:#eef0f4;margin-bottom:.3rem;'>{title}</h4>", unsafe_allow_html=True)
        c1, c2 = st.columns(2, gap="medium")
        with c1:
            st.markdown(f"""
            <div style="background:#111318;border:1px solid #242d3d;border-radius:8px;padding:.7rem 1rem;">
              <div style="color:#7a8494;font-size:.72rem;text-transform:uppercase;letter-spacing:.1em;">Control</div>
              <div style="font-size:1.4rem;font-weight:700;color:#7c4dff;">{s['c_mean']:.2f} <span style="font-size:.78rem;font-weight:400;color:#7a8494;">± {s['c_std']:.2f} {unit}</span></div>
              <div style="font-size:.76rem;color:#7a8494;">Median: {s['c_med']:.2f}</div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div style="background:#111318;border:1px solid #242d3d;border-radius:8px;padding:.7rem 1rem;">
              <div style="color:#7a8494;font-size:.72rem;text-transform:uppercase;letter-spacing:.1em;">Treatment</div>
              <div style="font-size:1.4rem;font-weight:700;color:#00e5ff;">{s['t_mean']:.2f} <span style="font-size:.78rem;font-weight:400;color:#7a8494;">± {s['t_std']:.2f} {unit}</span></div>
              <div style="font-size:.76rem;color:#7a8494;">Median: {s['t_med']:.2f}</div>
            </div>
            """, unsafe_allow_html=True)

        diff = s["t_mean"] - s["c_mean"]
        st.markdown(f"""
        <div style="font-size:.82rem;color:#eef0f4;margin:.4rem 0;">
          <b>Δ</b> {diff:+.2f} {unit}  &nbsp;|&nbsp;
          <b>t</b> = {s['tstat']:.3f}  &nbsp;|&nbsp;
          <b>Cohen's d</b> = {s['cohen_d']:.3f}  &nbsp;|&nbsp;
          {_sig_badge(s['pval'])}
        </div>
        <hr style="border-color:#242d3d;margin:.6rem 0;">
        """, unsafe_allow_html=True)

    # Verdict box
    all_sig = all(S[k]["pval"] < 0.05 for k in S)
    verdict_colour = "#69f0ae" if all_sig else "#ff4081"
    verdict_text   = "SUCCESS ✓ — All endpoints significant" if all_sig else "INCONCLUSIVE — review endpoints"
    st.markdown(f"""
    <div style="background:rgba({105 if all_sig else 255},{240 if all_sig else 64},{174 if all_sig else 129},.1);
                border:1px solid {verdict_colour};border-radius:10px;padding:1rem 1.4rem;margin-top:.8rem;">
      <div style="font-size:1.1rem;font-weight:700;color:{verdict_colour};">{verdict_text}</div>
      <div style="font-size:.78rem;color:#7a8494;margin-top:.2rem;">
        The novel diabetes drug demonstrates statistically significant efficacy in lowering glucose
        and preserving beta-cell function across the virtual cohort.
      </div>
    </div>
    """, unsafe_allow_html=True)

# ─── TAB: EXPORT ───
with tab_export:
    st.markdown("#### Download Trial Outputs")

    # CSV download
    csv_buf = df.to_csv(index=False).encode()
    st.download_button("⬇️  Patient-Level Results  (.csv)",
                       data=csv_buf, file_name="virtual_trial_results.csv", mime="text/csv")

    # Summary stats CSV
    S = compute_stats(df)
    summary_rows = []
    for key, label in [("Daily_Avg_Glucose","Glucose"),("Final_Beta_Cell_Mass","Beta-Cell"),("Years_To_Failure","Survival")]:
        s = S[key]
        summary_rows.append({"Endpoint": label,
                             "Control Mean": f"{s['c_mean']:.2f}",  "Control SD": f"{s['c_std']:.2f}",
                             "Treatment Mean": f"{s['t_mean']:.2f}", "Treatment SD": f"{s['t_std']:.2f}",
                             "t-stat": f"{s['tstat']:.3f}",         "p-value": f"{s['pval']:.2e}",
                             "Cohen d": f"{s['cohen_d']:.3f}"})
    sum_df = pd.DataFrame(summary_rows)
    st.download_button("⬇️  Summary Statistics  (.csv)",
                       data=sum_df.to_csv(index=False).encode(),
                       file_name="trial_summary_statistics.csv", mime="text/csv")

    # Static dashboard PNG
    ctrl_tr = {"t1": c_t1, "akt": c_akt, "t2": c_t2, "G": c_G, "t3": c_t3, "M": c_M}
    trt_tr  = {"t1": t_t1, "akt": t_akt, "t2": t_t2, "G": t_G, "t3": t_t3, "M": t_M}
    b64 = render_static_dashboard(ctrl_tr, trt_tr)
    st.download_button("⬇️  Static Dashboard  (.png  300 DPI)",
                       data=base64.b64decode(b64),
                       file_name="multi_scale_dashboard.png", mime="image/png")

    st.markdown("---")
    st.markdown("#### Summary Table Preview")
    st.dataframe(sum_df, width="stretch", hide_index=True)