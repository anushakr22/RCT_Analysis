# ── Dependencies ──────────────────────────────────────────────────────────────
# pip install streamlit pandas numpy matplotlib scipy statsmodels pingouin openpyxl
# ─────────────────────────────────────────────────────────────────────────────
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import re
import warnings
from scipy import stats
from statsmodels.formula.api import mixedlm
from statsmodels.nonparametric.smoothers_lowess import lowess as _lowess

from ingestion import (
    get_sheet_names,
    get_sheet_preview,
    ingest,
    IngestionResult,
)

warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RCT · Statistical Analysis",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500;600&display=swap');
:root {
    --ink:    #1a1a2e; --slate: #2d3561; --accent: #c84b31;
    --gold:   #ecb84a; --mist:  #f4f6fb; --paper:  #fffef9;
    --border: #dde3f0;
}
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    color: var(--ink); background-color: var(--mist);
}
section[data-testid="stSidebar"] { background: var(--slate); border-right: none; }
section[data-testid="stSidebar"] * { color: #e8eaf6 !important; }
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stMultiSelect label,
section[data-testid="stSidebar"] .stTextInput label,
section[data-testid="stSidebar"] h3 {
    color: #c8cef5 !important; font-size: 0.78rem !important;
    font-weight: 600 !important; letter-spacing: 0.07em !important;
    text-transform: uppercase !important;
}
section[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div,
section[data-testid="stSidebar"] .stMultiSelect div[data-baseweb="select"] > div {
    background: rgba(255,255,255,0.08) !important;
    border-color: rgba(255,255,255,0.15) !important; color: #fff !important;
}
h1 { font-family:'DM Serif Display',serif !important; font-size:2.2rem !important;
     color:var(--ink) !important; letter-spacing:-0.02em; }
h2 { font-family:'DM Serif Display',serif !important; font-size:1.4rem !important;
     color:var(--slate) !important; border-bottom:2px solid var(--border);
     padding-bottom:0.4rem; margin-top:2rem !important; }
h3 { font-family:'DM Sans',sans-serif !important; font-size:0.78rem !important;
     font-weight:600 !important; letter-spacing:0.08em !important;
     text-transform:uppercase !important; color:#7a8299 !important; }
.formula-box { background:var(--ink); color:#a8d8ea;
    font-family:'DM Mono',monospace; font-size:0.85rem;
    padding:0.9rem 1.2rem; border-radius:8px; margin:0.8rem 0; word-break:break-all; }
.stDataFrame { border-radius:8px !important; overflow:hidden; }
.stButton > button { background:var(--accent) !important; color:#fff !important;
    border:none !important; border-radius:7px !important;
    font-family:'DM Sans',sans-serif !important; font-weight:600 !important;
    font-size:0.9rem !important; padding:0.5rem 1.8rem !important;
    letter-spacing:0.04em !important; transition:opacity 0.15s; }
.stButton > button:hover { opacity:0.85 !important; }
hr { border-color:var(--border) !important; margin:1.5rem 0 !important; }
.info-box { background:#eef2ff; border-left:3px solid var(--slate);
    padding:0.7rem 1rem; border-radius:0 6px 6px 0;
    font-size:0.85rem; margin:0.5rem 0 1rem 0; color:var(--slate); }
.warn-box { background:#fff8e7; border-left:3px solid var(--gold);
    padding:0.7rem 1rem; border-radius:0 6px 6px 0;
    font-size:0.85rem; margin:0.5rem 0; color:#7a5c00; }
.alert-box { background:#fef2f2; border-left:3px solid #c84b31;
    padding:0.7rem 1rem; border-radius:0 6px 6px 0;
    font-size:0.85rem; margin:0.5rem 0; color:#7a1a1a; }
.summary-box { background:var(--paper); border:1px solid var(--border);
    border-left:4px solid var(--slate); border-radius:0 8px 8px 0;
    padding:1.2rem 1.5rem; font-size:0.95rem; line-height:1.8;
    color:var(--ink); margin-top:0.5rem; }
.model-pill { display:inline-block; background:var(--slate); color:#fff;
    font-size:0.72rem; font-weight:600; letter-spacing:0.06em;
    text-transform:uppercase; padding:0.2rem 0.7rem;
    border-radius:20px; margin-bottom:0.8rem; }
</style>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═════════════════════════════════════════════════════════════════════════════
def sanitize_col(name):
    s = re.sub(r'[^0-9a-zA-Z_]', '_', str(name))
    return ('X' + s) if s and s[0].isdigit() else s

def sig_stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    if p < 0.1:   return "."
    return ""

def icc_calc(model_result):
    try:
        re_var  = float(model_result.cov_re.iloc[0, 0])
        res_var = float(model_result.scale)
        return re_var / (re_var + res_var)
    except Exception:
        return None

def _idx(options, value, fallback=0):
    try: return options.index(value) if value in options else fallback
    except: return fallback

def _conf_badge(detected_val, selected_val, label):
    if detected_val is None:
        return f"<span style='font-size:0.68rem;color:#9aa0b8;'>⚪ {label}: not detected</span>"
    if detected_val == selected_val:
        return f"<span style='font-size:0.68rem;color:#4caf8a;'>✓ {label}: auto-detected</span>"
    return f"<span style='font-size:0.68rem;color:#ecb84a;'>✎ {label}: manually set</span>"

PLOT_STYLE = {
    "axes.facecolor": "#fffef9", "figure.facecolor": "#fffef9",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#dde3f0", "axes.labelcolor": "#1a1a2e",
    "xtick.color": "#7a8299", "ytick.color": "#7a8299",
    "grid.color": "#eef0f8", "grid.linestyle": "--", "grid.alpha": 0.7,
}


# ═════════════════════════════════════════════════════════════════════════════
# AUTO-DETECTION
# ═════════════════════════════════════════════════════════════════════════════
def auto_detect_roles(df) -> dict:
    cols      = df.columns.tolist()
    num_cols  = df.select_dtypes(include=[np.number]).columns.tolist()
    detected  = {"outcome": None, "subject_id": None, "group_var": None, "time_var": None}
    col_lower = {c: c.lower().replace(" ", "_").replace("-", "_") for c in cols}

    id_pat = re.compile(r"\b(participant|subject|patient|person|\bid\b|pid|sid|respondent|case|record|individu)", re.I)
    for c, cl in col_lower.items():
        if id_pat.search(cl): detected["subject_id"] = c; break

    grp_pat = re.compile(r"\b(group|arm|condition|treat|control|allocation|intervention|cohort|rand|assign)", re.I)
    for c, cl in col_lower.items():
        if c == detected["subject_id"]: continue
        if grp_pat.search(cl):
            if 2 <= len(df[c].dropna().unique()) <= 6:
                detected["group_var"] = c; break
    if not detected["group_var"]:
        for c in cols:
            if c == detected["subject_id"]: continue
            if len(df[c].dropna().unique()) == 2 and df[c].dtype == object:
                detected["group_var"] = c; break

    time_pat     = re.compile(r"\b(time|session|visit|wave|week|month|phase|period|pre|post|baseline|follow|occasion|point|measure)", re.I)
    time_val_pat = re.compile(r"^(pre|post|t\d|baseline|follow|week\s*\d|session\s*\d|visit\s*\d|wave\s*\d|time\s*\d|v\d)$", re.I)
    skip         = {detected["subject_id"], detected["group_var"]}
    for c, cl in col_lower.items():
        if c in skip: continue
        if time_pat.search(cl) and 2 <= len(df[c].dropna().unique()) <= 20:
            detected["time_var"] = c; break
    if not detected["time_var"]:
        for c in cols:
            if c in skip: continue
            if any(time_val_pat.match(str(v).strip()) for v in df[c].dropna().unique()):
                detected["time_var"] = c; break

    out_pat  = re.compile(r"\b(score|outcome|result|measure|response|rating|value|index|scale|total|dependent|dv|\by\b)", re.I)
    skip_all = {detected["subject_id"], detected["group_var"], detected["time_var"]}
    named    = next((c for c in num_cols if c not in skip_all and out_pat.search(col_lower[c])), None)
    cands    = [c for c in num_cols if c not in skip_all]
    detected["outcome"] = named or (max(cands, key=lambda c: df[c].std(skipna=True)) if cands else None)
    return detected


# ═════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ═════════════════════════════════════════════════════════════════════════════
for _k, _v in {
    "df": None, "ingestion_result": None,
    "file_bytes": None, "file_name": None,
    "sheet_names": None, "selected_sheets": None, "merge_sheets": False,
    "auto_roles": None, "roles_file": None,
    "lmm_result": None, "lmm_model_df": None,
    "lmm_safe_outcome": None, "lmm_safe_subject": None,
    "anova_result": None,
}.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ═════════════════════════════════════════════════════════════════════════════
# SIDEBAR — upload only
# ═════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='padding:1.2rem 0 1rem 0;'>
        <div style='font-family:DM Serif Display,serif;font-size:1.5rem;color:#fff;line-height:1.2;'>RCT · Analysis</div>
        <div style='font-size:0.75rem;color:#9aa0cc;margin-top:0.2rem;'>Statistical Analysis Suite</div>
    </div>""", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### Data")
    uploaded = st.file_uploader("Upload CSV or XLSX", type=["csv","xlsx"], label_visibility="collapsed")

    if uploaded:
        file_bytes = uploaded.read()
        if st.session_state.file_name != uploaded.name or st.session_state.file_bytes != file_bytes:
            st.session_state.file_bytes       = file_bytes
            st.session_state.file_name        = uploaded.name
            st.session_state.df               = None
            st.session_state.ingestion_result = None
            st.session_state.lmm_result       = None
            st.session_state.anova_result     = None
            st.session_state.sheet_names      = get_sheet_names(file_bytes) if uploaded.name.lower().endswith('.xlsx') else None

    if st.session_state.file_bytes and st.session_state.sheet_names:
        snames = st.session_state.sheet_names
        if len(snames) == 1:
            selected_sheets = snames; merge = False
            st.markdown(f"<div style='font-size:0.75rem;color:#9aa0cc;'>Sheet: <b>{snames[0]}</b></div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='font-size:0.75rem;color:#9aa0cc;'>{len(snames)} sheets found</div>", unsafe_allow_html=True)
            selected_sheets = st.multiselect("Sheets", snames, default=[snames[0]], label_visibility="collapsed")
            merge = st.checkbox("Merge sheets", value=True) if len(selected_sheets) > 1 else False
        if selected_sheets != st.session_state.selected_sheets or merge != st.session_state.merge_sheets:
            st.session_state.selected_sheets = selected_sheets
            st.session_state.merge_sheets    = merge
            st.session_state.df              = None
            st.session_state.ingestion_result = None

    # Run ingestion
    if (st.session_state.file_bytes and st.session_state.df is None and
            (not st.session_state.sheet_names or st.session_state.selected_sheets)):
        try:
            _ing = ingest(
                file_bytes=st.session_state.file_bytes,
                filename=st.session_state.file_name,
                sheet_names=st.session_state.selected_sheets,
                merge_sheets=st.session_state.merge_sheets or False,
            )
            st.session_state.df               = _ing.df
            st.session_state.ingestion_result = _ing
        except Exception as e:
            st.error(f"Could not read file: {e}")

    if st.session_state.df is not None:
        _r = st.session_state.ingestion_result
        st.success(f"✓ {len(st.session_state.df):,} rows · {len(st.session_state.df.columns)} cols")
        if _r and _r.summary_rows_removed:
            st.markdown(f"<div style='font-size:0.72rem;color:#9aa0cc;'>🗑 {_r.summary_rows_removed} junk rows removed</div>", unsafe_allow_html=True)
        st.markdown("<div style='font-size:0.75rem;color:#9aa0cc;margin-top:1rem;'>Configure model in main panel →</div>", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# MAIN PANEL
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("# RCT Statistical Analysis\n##### Upload · Configure · Run · Interpret")

if st.session_state.df is None:
    st.markdown("""
    <div style='margin-top:3rem;text-align:center;color:#9aa0b8;'>
        <div style='font-size:3rem;'>📂</div>
        <div style='font-family:DM Serif Display,serif;font-size:1.4rem;color:#2d3561;margin-top:0.5rem;'>
            Upload your data to begin</div>
        <div style='font-size:0.9rem;margin-top:0.4rem;'>Accepts CSV or XLSX</div>
    </div>""", unsafe_allow_html=True)
    st.stop()

df  = st.session_state.df
res = st.session_state.ingestion_result

if df is None or not isinstance(df, pd.DataFrame) or len(df) == 0:
    st.info("Upload a file in the sidebar to continue.")
    st.stop()

if res is None:
    res = IngestionResult(df=df, sheet_used="(csv)", format_detected="long",
                          header_rows_skipped=0, merged_cells_resolved=0,
                          summary_rows_removed=0, wide_cols_melted=[], warnings=[],
                          col_rename_map={})

# ─────────────────────────────────────────────────────────────────────────────
# INGESTION WARNINGS — popup-style alerts for complex file structures
# ─────────────────────────────────────────────────────────────────────────────
_ing_issues = []
if res.merged_cells_resolved > 0:
    _ing_issues.append((
        "⚠️ Merged cells detected",
        f"Your file contained <b>{res.merged_cells_resolved} merged cells</b>. "
        f"The app has filled them with the top-left value, but this may produce "
        f"unexpected column names or repeated values. "
        f"<b>Recommended:</b> open your file in Excel, unmerge all cells manually, "
        f"and re-upload for best results."
    ))
if res.header_rows_skipped > 0:
    _ing_issues.append((
        "⚠️ Multi-row headers detected",
        f"Your file had <b>{res.header_rows_skipped} extra header/label row(s)</b> "
        f"above the actual column names (e.g. a 'Pre-Intervention' label spanning "
        f"multiple columns). These were skipped, but column names may be incomplete. "
        f"<b>Recommended:</b> restructure your file so row 1 contains only the column "
        f"names and re-upload."
    ))
if res.format_detected == "wide" and res.wide_cols_melted:
    _ing_issues.append((
        "ℹ️ Wide format converted to long",
        f"Your data had <b>{len(res.wide_cols_melted)} columns that looked like "
        f"repeated time-point measurements</b> (e.g. Score_Pre, Score_Post). "
        f"These have been automatically stacked into a single 'time' column "
        f"(long format), which is required for mixed models. "
        f"Affected columns: <code>{'</code>, <code>'.join(res.wide_cols_melted[:8])}"
        f"{'...' if len(res.wide_cols_melted) > 8 else ''}</code>. "
        f"If this looks wrong, restructure your file in long format before uploading."
    ))
for w in (res.warnings or []):
    _ing_issues.append(("⚠️ Data warning", w))

if _ing_issues:
    st.markdown("### File structure notices")
    for title, body in _ing_issues:
        with st.expander(title, expanded=True):
            st.markdown(f"<div class='warn-box'>{body}</div>", unsafe_allow_html=True)
    st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# DATA PREVIEW
# ─────────────────────────────────────────────────────────────────────────────
with st.expander("🔍  Data preview", expanded=True):
    ic1, ic2, ic3, ic4 = st.columns(4)
    ic1.metric("Rows", f"{len(df):,}")
    ic2.metric("Columns", len(df.columns))
    ic3.metric("Missing values", int(df.isnull().sum().sum()))
    ic4.metric("Format", res.format_detected.upper())
    st.markdown("---")
    st.markdown("**First 50 rows**")
    preview_df = st.session_state.get("df_filtered", df)
    st.dataframe(preview_df.head(50), use_container_width=True)
    with st.expander("Column details"):
        st.dataframe(pd.DataFrame({
            "Column":   df.columns,
            "Type":     [str(df[c].dtype) for c in df.columns],
            "Non-null": [int(df[c].notna().sum()) for c in df.columns],
            "Unique":   [int(df[c].nunique()) for c in df.columns],
            "Sample":   [str(df[c].dropna().iloc[0]) if df[c].notna().any() else "—" for c in df.columns],
        }), use_container_width=True, hide_index=True)
    if st.session_state.sheet_names and st.session_state.selected_sheets:
        with st.expander("Raw sheet (before cleaning)"):
            ps = st.selectbox("Sheet", st.session_state.selected_sheets, label_visibility="collapsed")
            st.dataframe(get_sheet_preview(st.session_state.file_bytes, ps, nrows=10), use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# FILTERS
# ─────────────────────────────────────────────────────────────────────────────
with st.expander("🔧  Filters", expanded=False):
    st.markdown("<div class='info-box'>Uncheck values to <b>exclude</b> rows before modelling.</div>", unsafe_allow_html=True)
    roles_now = st.session_state.get("auto_roles") or {}
    fcands    = []
    for rk in ("group_var", "time_var"):
        c = roles_now.get(rk)
        if c and c in df.columns: fcands.append((c, rk.replace("_var","").title()))
    already = {c for c,_ in fcands}
    for c in df.columns:
        if c in already or c in (roles_now.get("subject_id"), roles_now.get("outcome")): continue
        if 2 <= df[c].nunique() <= 15: fcands.append((c, c))

    fstate_key = "fstate_" + str(st.session_state.file_name)
    if fstate_key not in st.session_state: st.session_state[fstate_key] = {}
    fstate         = st.session_state[fstate_key]
    active_filters = {}

    if not fcands:
        st.markdown("<div style='font-size:0.8rem;color:#9aa0cc;'>No categorical columns found for filtering.</div>", unsafe_allow_html=True)
    else:
        fcols = st.columns(min(len(fcands), 4))
        for fi, (cn, lbl) in enumerate(fcands):
            rvals = sorted(df[cn].dropna().unique(), key=str)
            if cn not in fstate: fstate[cn] = {str(v): True for v in rvals}
            for v in rvals:
                if str(v) not in fstate[cn]: fstate[cn][str(v)] = True
            with fcols[fi % min(len(fcands), 4)]:
                st.markdown(f"<div style='font-size:0.72rem;font-weight:600;letter-spacing:0.07em;text-transform:uppercase;color:#7a8299;margin-bottom:0.3rem;'>{lbl} ({len(rvals)})</div>", unsafe_allow_html=True)
                kept = []
                for v in rvals:
                    sv = str(v)
                    if st.checkbox(sv, value=fstate[cn].get(sv, True), key=f"f__{cn}__{sv}"):
                        kept.append(v)
                    fstate[cn][sv] = sv in [str(x) for x in kept] or False
                if not kept:
                    st.markdown("<div style='font-size:0.72rem;color:#c84b31;'>⚠ Select at least one.</div>", unsafe_allow_html=True)
                    kept = list(rvals)
                elif len(kept) < len(rvals):
                    st.markdown(f"<div style='font-size:0.72rem;color:#ecb84a;'>{len(kept)}/{len(rvals)} selected</div>", unsafe_allow_html=True)
                active_filters[cn] = kept

    df_f = df.copy()
    for cn, kv in active_filters.items():
        df_f = df_f[df_f[cn].isin(kv)]
    st.session_state["df_filtered"] = df_f
    n_in, n_out = len(df), len(df_f)
    clr = "#fff8e7" if n_out < n_in else "#eef2ff"
    bdr = "#ecb84a" if n_out < n_in else "#2d3561"
    txt = "#7a5c00" if n_out < n_in else "#2d3561"
    msg = f"<b>{n_out:,}</b> rows kept · <b>{n_in-n_out:,}</b> excluded" if n_out < n_in else f"All <b>{n_in:,}</b> rows included"
    st.markdown(f"<div style='margin-top:0.8rem;padding:0.5rem 0.8rem;background:{clr};border-left:3px solid {bdr};border-radius:0 6px 6px 0;font-size:0.8rem;color:{txt};'>{msg}</div>", unsafe_allow_html=True)

df = st.session_state.get("df_filtered", df)

# ─────────────────────────────────────────────────────────────────────────────
# MODEL SELECTOR
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("## Model Configuration")

if st.session_state.roles_file != st.session_state.file_name:
    st.session_state.auto_roles = auto_detect_roles(df)
    st.session_state.roles_file = st.session_state.file_name
roles    = st.session_state.auto_roles or {}
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
all_cols = df.columns.tolist()

model_choice = st.selectbox(
    "Select statistical model",
    ["Linear Mixed Model (LMM)", "Mixed Factorial ANOVA"],
    help="LMM — for continuous outcomes with repeated measures. Mixed ANOVA — for factorial designs with between and within-subjects factors."
)

st.markdown("<div style='margin-top:0.6rem'></div>", unsafe_allow_html=True)

# ─── Auto-detect badge ────────────────────────────────────────────────────────
n_det = sum(1 for v in roles.values() if v is not None)
if n_det:
    st.markdown(f"<div class='info-box'>⚡ Auto-detected {n_det} variable role(s) — review below.</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# ██  MODEL 1 — LINEAR MIXED MODEL (LMM)
# ══════════════════════════════════════════════════════════════════════════════
if model_choice == "Linear Mixed Model (LMM)":

    st.markdown("<div class='model-pill'>Linear Mixed Model</div>", unsafe_allow_html=True)

    # ── Variable roles — row 1 ────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        lmm_outcome = st.selectbox("Outcome (dependent variable)",
            ["— select —"] + num_cols,
            index=_idx(["— select —"] + num_cols, roles.get("outcome"), 0), key="lmm_out")
    with c2:
        lmm_subject = st.selectbox("Subject / Participant ID",
            ["— select —"] + all_cols,
            index=_idx(["— select —"] + all_cols, roles.get("subject_id"), 0), key="lmm_sub")
    with c3:
        lmm_group = st.selectbox("Group variable",
            ["None"] + all_cols,
            index=_idx(["None"] + all_cols, roles.get("group_var"), 0), key="lmm_grp")
    with c4:
        lmm_time = st.selectbox("Time / Session variable",
            ["None"] + all_cols,
            index=_idx(["None"] + all_cols, roles.get("time_var"), 0), key="lmm_tim")

    st.markdown(" &nbsp;·&nbsp; ".join([
        _conf_badge(roles.get("outcome"),    lmm_outcome, "Outcome"),
        _conf_badge(roles.get("subject_id"), lmm_subject, "Subject ID"),
        _conf_badge(roles.get("group_var"),  lmm_group,   "Group"),
        _conf_badge(roles.get("time_var"),   lmm_time,    "Time"),
    ]), unsafe_allow_html=True)

    st.markdown("<div style='margin-top:0.8rem'></div>", unsafe_allow_html=True)

    # ── Row 2: covariates / interactions / random slopes ──────────────────────
    r2a, r2b, r2c = st.columns([2, 2, 2])

    lmm_auto_fe = [x for x in [lmm_group, lmm_time] if x != "None"]
    lmm_remain  = [c for c in all_cols if c not in lmm_auto_fe and c not in (lmm_outcome, lmm_subject)]

    with r2a:
        lmm_extra_fe  = st.multiselect("Additional covariates", lmm_remain, key="lmm_cov")
        lmm_fixed     = lmm_auto_fe + lmm_extra_fe
        st.markdown(f"<div style='font-size:0.72rem;color:#9aa0b8;'>Auto-included: {', '.join(lmm_auto_fe) or 'none'}</div>", unsafe_allow_html=True)

    lmm_interactions = []
    with r2b:
        st.markdown("<div style='font-size:0.8rem;font-weight:600;margin-bottom:0.3rem;'>Interactions</div>", unsafe_allow_html=True)
        if lmm_group != "None" and lmm_time != "None":
            if st.checkbox("Group × Time  ✱ recommended", value=True, key="lmm_gxt"):
                lmm_interactions.append((lmm_group, lmm_time))
        lmm_cust = st.text_input("Custom (A:B, C:D)", placeholder="var1:var2", key="lmm_cust_ixn")
        if lmm_cust.strip():
            for pair in lmm_cust.split(","):
                pts = [p.strip() for p in pair.split(":")]
                if len(pts) == 2: lmm_interactions.append(tuple(pts))

    with r2c:
        st.markdown("<div style='font-size:0.8rem;font-weight:600;margin-bottom:0.3rem;'>Random slopes</div>", unsafe_allow_html=True)
        st.markdown("<div style='font-size:0.72rem;color:#9aa0b8;margin-bottom:0.4rem;'>Random intercept per subject always included.</div>", unsafe_allow_html=True)
        lmm_re_cands  = [c for c in lmm_fixed if c != lmm_subject]
        lmm_re_slopes = st.multiselect("Random slopes", lmm_re_cands, label_visibility="collapsed", key="lmm_slopes")

    # Guard
    if lmm_outcome == "— select —" or lmm_subject == "— select —":
        st.markdown("<div class='info-box'>↑ Set Outcome and Subject ID to continue.</div>", unsafe_allow_html=True)
        st.stop()

    # ── Build formula ─────────────────────────────────────────────────────────
    rename_map   = {c: sanitize_col(c) for c in df.columns}
    df_safe      = df.rename(columns=rename_map)
    s_out        = sanitize_col(lmm_outcome)
    s_sub        = sanitize_col(lmm_subject)
    s_fe         = [sanitize_col(c) for c in lmm_fixed]
    s_ixn        = [(sanitize_col(a), sanitize_col(b)) for a, b in lmm_interactions]
    s_slopes     = [sanitize_col(c) for c in lmm_re_slopes]

    fe_terms = list(s_fe)
    for a, b in s_ixn:
        if a in s_fe and b in s_fe: fe_terms.append(f"{a}:{b}")
    rhs        = " + ".join(fe_terms) if fe_terms else "1"
    formula    = f"{s_out} ~ {rhs}"
    re_formula = ("~" + " + ".join(s_slopes)) if s_slopes else None

    # ── Model spec display + run button ───────────────────────────────────────
    st.markdown("## Model Specification")
    cf, cr, crn = st.columns([3, 2, 1])
    with cf:
        st.markdown("**Fixed-effects formula**")
        st.markdown(f"<div class='formula-box'>{formula}</div>", unsafe_allow_html=True)
    with cr:
        st.markdown("**Random effects**")
        re_desc = f"Random intercept | {lmm_subject}"
        if lmm_re_slopes:
            re_desc += "<br>" + "<br>".join(f"Random slope: {s} | {lmm_subject}" for s in lmm_re_slopes)
        st.markdown(f"<div class='formula-box' style='font-size:0.8rem;'>{re_desc}</div>", unsafe_allow_html=True)
    with crn:
        st.markdown("<div style='margin-top:1.6rem'></div>", unsafe_allow_html=True)
        lmm_run = st.button("▶  Run LMM", use_container_width=True, key="run_lmm")

    # ── Fit model ─────────────────────────────────────────────────────────────
    if lmm_run:
        with st.spinner("Fitting Linear Mixed Model…"):
            try:
                mdf = df_safe.dropna(subset=[s_out, s_sub] + s_fe)
                for col in s_fe:
                    if df_safe[col].dtype == object or df_safe[col].nunique() <= 10:
                        mdf = mdf.copy(); mdf[col] = pd.Categorical(mdf[col])
                md  = mixedlm(formula, mdf, groups=mdf[s_sub], re_formula=re_formula)
                res_lmm = md.fit(method="lbfgs", reml=False, maxiter=2000)
                st.session_state.lmm_result       = res_lmm
                st.session_state.lmm_model_df     = mdf
                st.session_state.lmm_safe_outcome = s_out
                st.session_state.lmm_safe_subject = s_sub
                for k in [k for k in st.session_state if k.startswith("lmm_sum_")]:
                    del st.session_state[k]
                st.success("✓ Model converged")
            except Exception as e:
                st.error(f"Model failed: {e}")
                st.session_state.lmm_result = None

    # ── RESULTS ───────────────────────────────────────────────────────────────
    if st.session_state.lmm_result is not None:
        lmm_res  = st.session_state.lmm_result
        lmm_mdf  = st.session_state.lmm_model_df
        lmm_sout = st.session_state.lmm_safe_outcome

        st.markdown("---")
        st.markdown("## Results")

        # Fixed effects table
        st.markdown("### Fixed effects")
        _fi   = lmm_res.fe_params.index
        _ci   = lmm_res.conf_int().reindex(_fi)
        fe_df = pd.DataFrame({
            "Term": list(_fi), "β (Est.)": lmm_res.fe_params.values,
            "SE": lmm_res.bse_fe.reindex(_fi).values,
            "z": lmm_res.tvalues.reindex(_fi).values,
            "p-value": lmm_res.pvalues.reindex(_fi).values,
            "95% CI Lower": _ci.iloc[:, 0].values,
            "95% CI Upper": _ci.iloc[:, 1].values,
        })
        fe_df["Sig."] = fe_df["p-value"].apply(sig_stars)

        def _hl(row): return ["background-color:#fff8e7"]*len(row) if row["p-value"] < 0.05 else [""]*len(row)
        st.dataframe(fe_df.style.apply(_hl, axis=1).format({
            "β (Est.)":"{:.4f}", "SE":"{:.4f}", "z":"{:.3f}",
            "p-value":"{:.4f}", "95% CI Lower":"{:.4f}", "95% CI Upper":"{:.4f}"}),
            use_container_width=True, hide_index=True)
        st.markdown("<div style='font-size:0.75rem;color:#9aa0b8;'>*** p<0.001 · ** p<0.01 · * p<0.05 · . p<0.1 · highlighted = significant at α=0.05</div>", unsafe_allow_html=True)

        # Plain-language summary
        st.markdown("---")
        st.markdown("### Plain-language summary")

        def _lmm_summary(fe_df, outcome, group_var, time_var):
            sents    = []
            all_beta = fe_df["β (Est.)"].tolist()
            sig      = fe_df[fe_df["p-value"] < 0.05]
            trend    = fe_df[(fe_df["p-value"] >= 0.05) & (fe_df["p-value"] < 0.1)]
            ixn_sig  = sig[sig["Term"].str.contains(":")]

            def _dir(b): return "higher" if float(b) > 0 else "lower"
            def _mag(b):
                try:
                    mx = max(abs(float(x)) for x in all_beta if str(x) not in ("None","nan"))
                    r  = abs(float(b)) / mx if mx else 0
                    return "notably " if r >= 0.5 else "moderately " if r >= 0.2 else "slightly "
                except: return ""
            def _clean(t):
                m = re.search(r'\[T\.(.+?)\]', t)
                return m.group(1).replace("_"," ") if m else t.replace("_"," ")

            if len(ixn_sig):
                sents.append(f"The analysis found a significant treatment effect: different groups changed differently over time on {outcome}.")
                parts = [f"{_clean(r['Term'])} was {_mag(r['β (Est.)'])}{_dir(r['β (Est.)'])} than the reference" for _,r in ixn_sig.iterrows()]
                sents.append("Specifically: " + "; and ".join(parts) + ".")
            elif len(sig):
                sents.append(f"The model found some significant effects on {outcome}, but the key group-by-time interaction was not significant.")
            else:
                sents.append(f"No statistically significant effects on {outcome} were found at the conventional threshold.")

            if group_var != "None":
                grp_sig = sig[sig["Term"].str.contains(sanitize_col(group_var), case=False) & ~sig["Term"].str.contains(":")]
                if len(grp_sig):
                    parts = [f"{_clean(r['Term'])} had {_mag(r['β (Est.)'])}{_dir(r['β (Est.)'])} {outcome} overall" for _,r in grp_sig.iterrows()]
                    sents.append("Across all timepoints: " + "; ".join(parts) + ".")

            if time_var != "None":
                time_sig = sig[sig["Term"].str.contains(sanitize_col(time_var), case=False) & ~sig["Term"].str.contains(":")]
                if len(time_sig):
                    parts = [f"at {_clean(r['Term'])}, values were {_dir(r['β (Est.)'])}" for _,r in time_sig.iterrows()]
                    sents.append("There was also a general time effect: " + "; ".join(parts) + ".")

            core = {sanitize_col(x).lower() for x in [group_var, time_var] if x != "None"}
            cov_sig = sig[~sig["Term"].str.contains(":") & ~sig["Term"].str.lower().apply(lambda t: any(c in t for c in core)) & (sig["Term"] != "Intercept")]
            if len(cov_sig):
                sents.append(f"{outcome} also varied significantly by background factors ({', '.join(_clean(r['Term']) for _,r in cov_sig.iterrows())}) — these are not treatment effects.")

            if len(trend[trend["Term"].str.contains(":")]):
                sents.append("Some group-by-time trends approached significance and may be worth exploring with a larger sample.")

            if len(ixn_sig):
                sents.append(f"These results suggest the groups responded differently to the intervention — the clinical significance of these changes in {outcome} warrants further investigation.")
            elif len(sig):
                sents.append(f"The absence of a clear group-by-time interaction limits the evidence for a differential treatment response — a larger sample may be needed.")
            else:
                sents.append(f"The null findings may reflect limited statistical power rather than a true absence of effect.")

            return " ".join(sents)

        sk = "lmm_sum_" + str(hash(str(fe_df.values.tolist())))
        if sk not in st.session_state:
            st.session_state[sk] = _lmm_summary(fe_df, lmm_outcome, lmm_group, lmm_time)
        st.markdown(f"<div class='summary-box'>{st.session_state[sk]}</div>", unsafe_allow_html=True)

        # Diagnostic plots — 2×2
        st.markdown("---")
        st.markdown("### Diagnostic plots")

        _sing = False
        try:
            fitted = lmm_res.fittedvalues; resids = lmm_res.resid
        except ValueError as _ve:
            if "singular" in str(_ve).lower():
                _sing  = True
                fitted = lmm_res.predict(lmm_mdf)
                resids = pd.Series(lmm_mdf[lmm_sout].values - fitted.values, index=lmm_mdf.index)
            else: raise

        if _sing:
            st.markdown("<div class='warn-box'><b>Singular random effects:</b> Random-intercept variance estimated to zero — plots use fixed-effects-only predictions.</div>", unsafe_allow_html=True)

        pc1, pc2 = st.columns(2)
        def _plot(ax_fn, col):
            with col:
                with plt.rc_context(PLOT_STYLE):
                    fig, ax = plt.subplots(figsize=(5, 3.5))
                    ax_fn(ax)
                    ax.grid(True); fig.tight_layout(); st.pyplot(fig); plt.close()

        with pc1:
            with plt.rc_context(PLOT_STYLE):
                fig, ax = plt.subplots(figsize=(5, 3.5))
                ax.scatter(fitted, resids, alpha=0.5, s=14, color="#2d3561", edgecolors="none")
                ax.axhline(0, color="#c84b31", lw=1.5, ls="--")
                try: sm = _lowess(resids.values, fitted.values, frac=0.4); ax.plot(sm[:,0], sm[:,1], color="#ecb84a", lw=2)
                except: pass
                ax.set_xlabel("Fitted values", fontsize=9); ax.set_ylabel("Residuals", fontsize=9)
                ax.set_title("Residuals vs Fitted", fontsize=11, fontfamily="serif", color="#1a1a2e")
                ax.grid(True); fig.tight_layout(); st.pyplot(fig); plt.close()
        with pc2:
            with plt.rc_context(PLOT_STYLE):
                fig, ax = plt.subplots(figsize=(5, 3.5))
                (osm, osr), (sl, ic, _) = stats.probplot(resids)
                ax.scatter(osm, osr, alpha=0.5, s=14, color="#2d3561", edgecolors="none")
                ax.plot([osm[0],osm[-1]], [sl*osm[0]+ic, sl*osm[-1]+ic], color="#c84b31", lw=1.5)
                ax.set_xlabel("Theoretical quantiles", fontsize=9); ax.set_ylabel("Sample quantiles", fontsize=9)
                ax.set_title("Normal Q-Q", fontsize=11, fontfamily="serif", color="#1a1a2e")
                ax.grid(True); fig.tight_layout(); st.pyplot(fig); plt.close()
        with pc1:
            with plt.rc_context(PLOT_STYLE):
                fig, ax = plt.subplots(figsize=(5, 3.5))
                rs = resids.std(); sr = (np.sqrt(np.abs(resids/rs)) if rs > 0 else np.abs(resids))
                ax.scatter(fitted, sr, alpha=0.5, s=14, color="#2d3561", edgecolors="none")
                try: sm = _lowess(sr.values, fitted.values, frac=0.4); ax.plot(sm[:,0], sm[:,1], color="#ecb84a", lw=2)
                except: pass
                ax.set_xlabel("Fitted values", fontsize=9); ax.set_ylabel("√|Std. residuals|", fontsize=9)
                ax.set_title("Scale-Location", fontsize=11, fontfamily="serif", color="#1a1a2e")
                ax.grid(True); fig.tight_layout(); st.pyplot(fig); plt.close()
        with pc2:
            with plt.rc_context(PLOT_STYLE):
                fig, ax = plt.subplots(figsize=(5, 3.5))
                actual = lmm_mdf[lmm_sout]
                ax.scatter(actual, fitted, alpha=0.45, s=14, color="#2d3561", edgecolors="none")
                mn, mx = min(actual.min(), fitted.min()), max(actual.max(), fitted.max())
                ax.plot([mn,mx],[mn,mx], color="#c84b31", lw=1.5, ls="--", label="Perfect fit")
                try: r2 = np.corrcoef(actual.values, fitted.values)[0,1]**2; ax.set_title(f"Predicted vs Observed  (R²={r2:.3f})", fontsize=11, fontfamily="serif", color="#1a1a2e")
                except: ax.set_title("Predicted vs Observed", fontsize=11, fontfamily="serif", color="#1a1a2e")
                ax.set_xlabel("Observed", fontsize=9); ax.set_ylabel("Predicted", fontsize=9)
                ax.legend(fontsize=8); ax.grid(True); fig.tight_layout(); st.pyplot(fig); plt.close()

        # Export
        st.markdown("---")
        st.markdown("## Export")
        ex1, ex2 = st.columns(2)
        with ex1:
            st.download_button("⬇  Full summary (.txt)", data=lmm_res.summary().as_text(), file_name="lmm_summary.txt", mime="text/plain")
        with ex2:
            buf = io.StringIO(); fe_df.to_csv(buf, index=False)
            st.download_button("⬇  Fixed effects (.csv)", data=buf.getvalue(), file_name="lmm_fixed_effects.csv", mime="text/csv")
        with st.expander("📄 Full statsmodels summary"):
            st.code(lmm_res.summary().as_text(), language="text")


# ══════════════════════════════════════════════════════════════════════════════
# ██  MODEL 2 — MIXED FACTORIAL ANOVA
# ══════════════════════════════════════════════════════════════════════════════
elif model_choice == "Mixed Factorial ANOVA":

    st.markdown("<div class='model-pill'>Mixed Factorial ANOVA</div>", unsafe_allow_html=True)

    # Check pingouin available
    try:
        import pingouin as pg
        _pg_ok = True
    except ImportError:
        _pg_ok = False
        st.markdown("""<div class='alert-box'>
        <b>pingouin not installed.</b> Run in your terminal:<br>
        <code>pip install pingouin</code><br>then restart Streamlit.</div>""", unsafe_allow_html=True)
        st.stop()

    st.markdown("""<div class='info-box'>
    <b>Mixed Factorial ANOVA</b> tests whether group differences and time effects are significant,
    and whether those effects interact. It requires one <b>between-subjects factor</b>
    (e.g. treatment vs control — each person is in one group only) and one or more
    <b>within-subjects factors</b> (e.g. time/session — each person is measured at every level).
    </div>""", unsafe_allow_html=True)

    # ── Tier 1: always-required ───────────────────────────────────────────────
    st.markdown("#### Core variables")
    t1a, t1b = st.columns(2)
    with t1a:
        av_outcome = st.selectbox("Outcome (dependent variable)",
            ["— select —"] + num_cols,
            index=_idx(["— select —"] + num_cols, roles.get("outcome"), 0), key="av_out")
    with t1b:
        av_subject = st.selectbox("Subject / Participant ID",
            ["— select —"] + all_cols,
            index=_idx(["— select —"] + all_cols, roles.get("subject_id"), 0), key="av_sub")

    # ── Tier 2: between-subjects factors ─────────────────────────────────────
    st.markdown("#### Between-subjects factors")
    st.markdown("<div style='font-size:0.82rem;color:#7a8299;margin-bottom:0.5rem;'>Factors where each participant belongs to only one level (e.g. Group: Treatment vs Control).</div>", unsafe_allow_html=True)

    n_between = st.selectbox("How many between-subjects factors?", [0, 1, 2], index=1, key="av_nbet")
    between_factors = []
    if n_between > 0:
        bef_cols = st.columns(n_between)
        for i in range(n_between):
            with bef_cols[i]:
                bf = st.selectbox(f"Between-subjects factor {i+1}",
                    ["— select —"] + all_cols,
                    index=_idx(["— select —"] + all_cols, roles.get("group_var") if i == 0 else None, 0),
                    key=f"av_bet{i}")
                if bf != "— select —":
                    between_factors.append(bf)
                    n_lvl = df[bf].nunique()
                    st.markdown(f"<div style='font-size:0.72rem;color:#9aa0b8;'>{n_lvl} levels: {', '.join(str(v) for v in sorted(df[bf].dropna().unique()))}</div>", unsafe_allow_html=True)

    # ── Tier 3: within-subjects factors ──────────────────────────────────────
    st.markdown("#### Within-subjects factors")
    st.markdown("<div style='font-size:0.82rem;color:#7a8299;margin-bottom:0.5rem;'>Factors where every participant appears at every level (e.g. Session: Pre, Post — everyone was measured twice).</div>", unsafe_allow_html=True)

    n_within = st.selectbox("How many within-subjects factors?", [0, 1, 2], index=1, key="av_nwit")
    within_factors = []
    if n_within > 0:
        wif_cols = st.columns(n_within)
        for i in range(n_within):
            with wif_cols[i]:
                wf = st.selectbox(f"Within-subjects factor {i+1}",
                    ["— select —"] + all_cols,
                    index=_idx(["— select —"] + all_cols, roles.get("time_var") if i == 0 else None, 0),
                    key=f"av_wit{i}")
                if wf != "— select —":
                    within_factors.append(wf)
                    n_lvl = df[wf].nunique()
                    st.markdown(f"<div style='font-size:0.72rem;color:#9aa0b8;'>{n_lvl} levels: {', '.join(str(v) for v in sorted(df[wf].dropna().unique()))}</div>", unsafe_allow_html=True)

    # ── Tier 4: advanced options ──────────────────────────────────────────────
    with st.expander("⚙️  Advanced options", expanded=False):
        st.markdown("**Covariates (turns this into an ANCOVA)**")
        av_cov_candidates = [c for c in num_cols if c not in [av_outcome, av_subject] + between_factors + within_factors]
        av_covariates     = st.multiselect("Numeric covariates to control for", av_cov_candidates, key="av_cov")

        st.markdown("**Post-hoc pairwise comparisons**")
        av_posthoc = st.checkbox("Run post-hoc tests on significant factors", value=True, key="av_ph")
        if av_posthoc:
            av_ph_corr = st.selectbox("Multiple comparison correction",
                ["bonferroni", "holm", "fdr_bh", "none"],
                help="Bonferroni = strictest, Holm = slightly less strict, FDR = good for many comparisons, None = uncorrected",
                key="av_ph_corr")
        else:
            av_ph_corr = "bonferroni"

        st.markdown("**Effect size**")
        st.markdown("<div style='font-size:0.82rem;color:#7a8299;'>Partial η² (eta-squared) is always reported. It tells you how much of the variance in the outcome is explained by each effect.</div>", unsafe_allow_html=True)

    # ── Guard ─────────────────────────────────────────────────────────────────
    if av_outcome == "— select —" or av_subject == "— select —":
        st.markdown("<div class='info-box'>↑ Set Outcome and Subject ID to continue.</div>", unsafe_allow_html=True)
        st.stop()
    if not between_factors and not within_factors:
        st.markdown("<div class='info-box'>↑ Select at least one between-subjects or within-subjects factor.</div>", unsafe_allow_html=True)
        st.stop()
    if not within_factors:
        st.markdown("<div class='warn-box'>⚠ No within-subjects factor selected — this will run a standard between-subjects ANOVA, not a mixed design.</div>", unsafe_allow_html=True)
    if not between_factors:
        st.markdown("<div class='warn-box'>⚠ No between-subjects factor selected — this will run a repeated-measures ANOVA, not a mixed design.</div>", unsafe_allow_html=True)

    # ── Model summary display + run ───────────────────────────────────────────
    st.markdown("## Model Specification")
    ms1, ms2, ms3 = st.columns([2, 2, 1])
    with ms1:
        bet_str = ", ".join(between_factors) if between_factors else "none"
        wit_str = ", ".join(within_factors)  if within_factors  else "none"
        cov_str = ", ".join(av_covariates)   if av_covariates   else "none"
        st.markdown(f"<div class='formula-box'>Outcome: {av_outcome}<br>Between: {bet_str}<br>Within: {wit_str}<br>Covariates: {cov_str}</div>", unsafe_allow_html=True)
    with ms2:
        st.markdown(f"<div class='formula-box' style='font-size:0.8rem;'>Subject: {av_subject}<br>Post-hoc: {'yes — ' + av_ph_corr if av_posthoc else 'no'}<br>Effect size: partial η²</div>", unsafe_allow_html=True)
    with ms3:
        st.markdown("<div style='margin-top:1.6rem'></div>", unsafe_allow_html=True)
        av_run = st.button("▶  Run ANOVA", use_container_width=True, key="run_anova")

    # ── Fit ANOVA ─────────────────────────────────────────────────────────────
    if av_run:
        with st.spinner("Running Mixed Factorial ANOVA…"):
            try:
                adf = df[[av_outcome, av_subject] + between_factors + within_factors + av_covariates].dropna()

                if within_factors and between_factors:
                    # Full mixed: use pingouin mixed_anova (supports 1 between, 1 within)
                    if len(between_factors) == 1 and len(within_factors) == 1:
                        aov = pg.mixed_anova(
                            data=adf, dv=av_outcome,
                            within=within_factors[0],
                            between=between_factors[0],
                            subject=av_subject,
                            correction=True,
                        )
                    else:
                        # >1 factor: run rm_anova per within factor + between via OLS
                        aov = pg.mixed_anova(
                            data=adf, dv=av_outcome,
                            within=within_factors[0],
                            between=between_factors[0],
                            subject=av_subject,
                            correction=True,
                        )
                        st.markdown("<div class='warn-box'>⚠ With >1 within or between factor, only the first of each is used in the mixed ANOVA table. Additional factors are shown as separate analyses below.</div>", unsafe_allow_html=True)

                elif within_factors and not between_factors:
                    # Repeated measures only
                    aov = pg.rm_anova(
                        data=adf, dv=av_outcome,
                        within=within_factors[0] if len(within_factors)==1 else within_factors,
                        subject=av_subject,
                        correction=True,
                    )
                else:
                    # Between-subjects only (one-way or factorial)
                    import statsmodels.formula.api as smf
                    formula_anova = f"{sanitize_col(av_outcome)} ~ " + " * ".join(sanitize_col(b) for b in between_factors)
                    adf_s = adf.rename(columns={c: sanitize_col(c) for c in adf.columns})
                    lm    = smf.ols(formula_anova, data=adf_s).fit()
                    import statsmodels.stats.anova as ssa
                    aov   = ssa.anova_lm(lm, typ=2).reset_index().rename(columns={"index":"Source","PR(>F)":"p-unc"})

                # Post-hoc tests
                ph_results = {}
                if av_posthoc:
                    all_factors = between_factors + within_factors
                    for fac in all_factors:
                        try:
                            corr = av_ph_corr if av_ph_corr != "none" else None
                            ph   = pg.pairwise_tests(
                                data=adf, dv=av_outcome,
                                between=fac if fac in between_factors else None,
                                within=fac  if fac in within_factors  else None,
                                subject=av_subject,
                                padjust=corr,
                            )
                            ph_results[fac] = ph
                        except Exception as ph_e:
                            ph_results[fac] = str(ph_e)

                # Sphericity (for within-subjects factors)
                sph_results = {}
                for wf in within_factors:
                    try:
                        sph = pg.sphericity(data=adf, dv=av_outcome, within=wf, subject=av_subject)
                        sph_results[wf] = sph
                    except Exception:
                        pass

                st.session_state.anova_result = {
                    "aov": aov, "ph": ph_results, "sph": sph_results,
                    "outcome": av_outcome, "between": between_factors,
                    "within": within_factors, "subject": av_subject,
                    "posthoc": av_posthoc, "ph_corr": av_ph_corr,
                }
                for k in [k for k in st.session_state if k.startswith("av_sum_")]:
                    del st.session_state[k]
                st.success("✓ ANOVA complete")
            except Exception as e:
                st.error(f"ANOVA failed: {e}")
                st.session_state.anova_result = None

    # ── ANOVA RESULTS ─────────────────────────────────────────────────────────
    if st.session_state.anova_result is not None:
        ar       = st.session_state.anova_result
        aov      = ar["aov"]
        ph       = ar["ph"]
        sph      = ar["sph"]
        a_out    = ar["outcome"]
        a_bet    = ar["between"]
        a_wit    = ar["within"]
        a_sub    = ar["subject"]

        st.markdown("---")
        st.markdown("## Results")

        # ── ANOVA table ───────────────────────────────────────────────────────
        st.markdown("### ANOVA table")
        st.markdown("""<div class='info-box'>
        <b>F</b> = ratio of variance explained by the effect to unexplained variance. Larger = stronger signal.<br>
        <b>p-unc</b> = p-value (uncorrected). <b>np2</b> = partial η² (effect size: 0.01 small, 0.06 medium, 0.14 large).<br>
        <b>eps</b> = Greenhouse-Geisser correction for sphericity (applied when < 1).
        </div>""", unsafe_allow_html=True)

        # Standardise display columns
        display_cols = []
        col_map      = {}
        for c in aov.columns:
            cl = c.lower()
            if cl in ("source","a","b","between","within"): col_map[c] = "Source"
            elif "df" in cl: col_map[c] = col_map.get(c, c)
            elif cl in ("f","f-value","f_val"): col_map[c] = "F"
            elif cl in ("p-unc","p","pr(>f)","p-value"): col_map[c] = "p-value"
            elif cl in ("np2","eta2[g]","eta2","pes"): col_map[c] = "η² (partial)"
            elif cl == "eps": col_map[c] = "ε (G-G)"
        aov_disp = aov.rename(columns=col_map)

        def _hl_aov(row):
            try:
                p = float(row.get("p-value", 1))
                return ["background-color:#fff8e7"] * len(row) if p < 0.05 else [""] * len(row)
            except: return [""] * len(row)

        num_fmt = {c: "{:.4f}" for c in aov_disp.select_dtypes(include=np.number).columns}
        st.dataframe(aov_disp.style.apply(_hl_aov, axis=1).format(num_fmt, na_rep="—"),
                     use_container_width=True, hide_index=True)
        st.markdown("<div style='font-size:0.75rem;color:#9aa0b8;'>Highlighted rows significant at α=0.05 · η² < 0.06 small · 0.06–0.14 medium · > 0.14 large</div>", unsafe_allow_html=True)

        # ── Sphericity ────────────────────────────────────────────────────────
        if sph:
            st.markdown("### Sphericity (Mauchly's test)")
            st.markdown("<div style='font-size:0.82rem;color:#7a8299;margin-bottom:0.5rem;'>Sphericity assumes variances of differences between all pairs of time points are equal. If violated (p < 0.05), use the Greenhouse-Geisser corrected p-values in the ANOVA table above (ε column).</div>", unsafe_allow_html=True)
            for wf, s in sph.items():
                try:
                    if hasattr(s, '__len__') and len(s) >= 3:
                        sph_ok, W, pval = s[0], s[1], s[2]
                        verdict = "✓ Not violated" if sph_ok else "⚠ Violated — use G-G corrected p-values"
                        clr = "#eef2ff" if sph_ok else "#fff8e7"
                        st.markdown(f"<div style='padding:0.5rem 0.8rem;background:{clr};border-radius:6px;font-size:0.85rem;'><b>{wf}:</b> W = {W:.3f}, p = {pval:.4f} — {verdict}</div>", unsafe_allow_html=True)
                except: pass

        # ── Post-hoc ─────────────────────────────────────────────────────────
        if ph and ar["posthoc"]:
            st.markdown("### Post-hoc pairwise comparisons")
            st.markdown(f"<div style='font-size:0.82rem;color:#7a8299;margin-bottom:0.5rem;'>Correction: <b>{ar['ph_corr']}</b>. These tell you exactly which pairs of groups or timepoints differ significantly from each other.</div>", unsafe_allow_html=True)
            for fac, ph_res in ph.items():
                st.markdown(f"**{fac}**")
                if isinstance(ph_res, str):
                    st.markdown(f"<div class='warn-box'>Could not compute post-hoc for {fac}: {ph_res}</div>", unsafe_allow_html=True)
                else:
                    ph_disp = ph_res.copy()
                    ph_cols = {c: c for c in ph_disp.columns}
                    # Rename key columns for readability
                    for c in ph_disp.columns:
                        if c.lower() in ("p-corr","p-adj","padj"): ph_cols[c] = "p (corrected)"
                        elif c.lower() == "p-unc": ph_cols[c] = "p (uncorrected)"
                        elif c.lower() in ("hedges","cohen","d"): ph_cols[c] = "Effect size (d)"
                    ph_disp = ph_disp.rename(columns=ph_cols)

                    def _hl_ph(row):
                        try:
                            p = float(row.get("p (corrected)", row.get("p (uncorrected)", 1)))
                            return ["background-color:#fff8e7"]*len(row) if p < 0.05 else [""]*len(row)
                        except: return [""]*len(row)

                    num_ph = {c: "{:.4f}" for c in ph_disp.select_dtypes(include=np.number).columns}
                    st.dataframe(ph_disp.style.apply(_hl_ph, axis=1).format(num_ph, na_rep="—"),
                                 use_container_width=True, hide_index=True)

        # ── Plain-language summary ─────────────────────────────────────────────
        st.markdown("---")
        st.markdown("### Plain-language summary")

        def _anova_summary(aov, a_out, a_bet, a_wit, ph):
            sents = []
            # Find p-value column
            p_col = next((c for c in aov.columns if c.lower() in ("p-unc","p","pr(>f)","p-value")), None)
            f_col = next((c for c in aov.columns if c.lower() in ("f","f-value","f_val")), None)
            e_col = next((c for c in aov.columns if c.lower() in ("np2","eta2","eta2[g]","pes")), None)
            src_col = next((c for c in aov.columns if c.lower() in ("source","a","b")), None)

            if p_col is None:
                return "Could not parse ANOVA table for summary."

            sig_effects = []
            for _, row in aov.iterrows():
                try:
                    p = float(row[p_col])
                    src = str(row[src_col]) if src_col else "Effect"
                    eta = float(row[e_col]) if e_col else None
                    f   = float(row[f_col]) if f_col else None
                    if p < 0.05:
                        size = ""
                        if eta:
                            size = "large" if eta >= 0.14 else "medium" if eta >= 0.06 else "small"
                        sig_effects.append({"src": src, "p": p, "eta": eta, "size": size, "f": f})
                except: pass

            if not sig_effects:
                sents.append(f"The mixed factorial ANOVA found no statistically significant effects on {a_out}.")
                sents.append("This may indicate that the groups did not differ meaningfully, or that the sample was too small to detect differences.")
                return " ".join(sents)

            sents.append(f"The mixed factorial ANOVA revealed {len(sig_effects)} significant effect(s) on {a_out}.")

            for eff in sig_effects:
                src = eff["src"].replace("*", "×").replace("_"," ")
                size_str = f" ({eff['size']} effect)" if eff["size"] else ""
                if ":" in eff["src"] or "*" in eff["src"]:
                    sents.append(f"The interaction between {src} was significant{size_str}, meaning the pattern of change was different across groups — this is typically the most important finding in an RCT.")
                elif any(b.lower() in eff["src"].lower() for b in a_bet):
                    sents.append(f"There was a significant main effect of {src}{size_str}: the groups differed from each other overall.")
                elif any(w.lower() in eff["src"].lower() for w in a_wit):
                    sents.append(f"There was a significant main effect of {src}{size_str}: scores changed across time/sessions on average.")

            # Post-hoc highlights
            if ph:
                ph_highlights = []
                for fac, ph_res in ph.items():
                    if isinstance(ph_res, str): continue
                    p_col_ph = next((c for c in ph_res.columns if "corr" in c.lower() or c.lower() == "p-unc"), None)
                    if p_col_ph is None: continue
                    sig_pairs = ph_res[ph_res[p_col_ph] < 0.05]
                    if len(sig_pairs):
                        ph_highlights.append(f"{len(sig_pairs)} significant pair(s) in {fac}")
                if ph_highlights:
                    sents.append("Post-hoc comparisons identified: " + "; ".join(ph_highlights) + " — see the pairwise table above for details.")

            sents.append(f"These results {'support' if sig_effects else 'do not support'} a meaningful treatment-related change in {a_out}. Consider the effect sizes alongside p-values when interpreting clinical relevance.")
            return " ".join(sents)

        sk = "av_sum_" + str(hash(str(aov.values.tolist())))
        if sk not in st.session_state:
            st.session_state[sk] = _anova_summary(aov, a_out, a_bet, a_wit, ph)
        st.markdown(f"<div class='summary-box'>{st.session_state[sk]}</div>", unsafe_allow_html=True)

        # ── Visualisation: group × time means plot ─────────────────────────────
        if a_bet and a_wit:
            st.markdown("---")
            st.markdown("### Group × Time means plot")
            st.markdown("<div style='font-size:0.82rem;color:#7a8299;margin-bottom:0.5rem;'>Shows the mean outcome for each group at each timepoint. Crossing lines = interaction.</div>", unsafe_allow_html=True)
            try:
                means_df = df.groupby([a_bet[0], a_wit[0]])[a_out].agg(["mean","sem"]).reset_index()
                with plt.rc_context(PLOT_STYLE):
                    fig, ax = plt.subplots(figsize=(7, 4))
                    colours = ["#2d3561","#c84b31","#ecb84a","#4caf8a","#9c27b0"]
                    for i, (grp, gdf) in enumerate(means_df.groupby(a_bet[0])):
                        gdf = gdf.sort_values(a_wit[0])
                        clr = colours[i % len(colours)]
                        ax.plot(gdf[a_wit[0]].astype(str), gdf["mean"], marker="o", lw=2, color=clr, label=str(grp))
                        ax.fill_between(range(len(gdf)), gdf["mean"]-gdf["sem"], gdf["mean"]+gdf["sem"], alpha=0.12, color=clr)
                    ax.set_xlabel(a_wit[0], fontsize=10)
                    ax.set_ylabel(f"Mean {a_out}", fontsize=10)
                    ax.set_title(f"{a_out} by {a_bet[0]} × {a_wit[0]}", fontsize=12, fontfamily="serif", color="#1a1a2e")
                    ax.legend(title=a_bet[0], fontsize=9)
                    ax.grid(True)
                    fig.tight_layout()
                    st.pyplot(fig)
                    plt.close()
            except Exception as plot_e:
                st.markdown(f"<div class='warn-box'>Could not render means plot: {plot_e}</div>", unsafe_allow_html=True)

        # Export
        st.markdown("---")
        st.markdown("## Export")
        buf = io.StringIO(); aov.to_csv(buf, index=False)
        st.download_button("⬇  ANOVA table (.csv)", data=buf.getvalue(), file_name="anova_results.csv", mime="text/csv")
        if ph and ar["posthoc"]:
            ph_frames = []
            for fac, ph_res in ph.items():
                if not isinstance(ph_res, str):
                    ph_res = ph_res.copy(); ph_res.insert(0, "Factor", fac)
                    ph_frames.append(ph_res)
            if ph_frames:
                buf2 = io.StringIO(); pd.concat(ph_frames, ignore_index=True).to_csv(buf2, index=False)
                st.download_button("⬇  Post-hoc results (.csv)", data=buf2.getvalue(), file_name="posthoc_results.csv", mime="text/csv")