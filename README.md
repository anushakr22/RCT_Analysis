# RCT · Linear Mixed Model Analyser

A Streamlit web app for non-technical statisticians to run Linear Mixed Models (LMM) on RCT data.

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the app
streamlit run app.py
```

The app will open in your browser at http://localhost:8501

---

## How to Use

### Step 1 — Upload Data
Upload a **CSV** or **XLSX** file. Data should be in **long format**:
- One row per observation per participant per time point
- Columns for: participant ID, group, time/session, outcome, and any covariates

### Step 2 — Variable Roles
In the sidebar, map your columns:
| Role | Example column name |
|------|-------------------|
| Outcome | `depression_score`, `pain_rating` |
| Subject ID | `participant_id`, `id` |
| Group | `group` (treatment / control) |
| Time / Session | `session` (pre / post), `week` |

### Step 3 — Fixed Effects
- Group and Time are auto-added as fixed effects
- Add any additional covariates (age, gender, baseline, etc.)
- Interactions: Group × Time is offered automatically; custom interactions use `var1:var2` syntax

### Step 4 — Random Effects
- Random intercept per subject is always included
- Optionally add random slopes (e.g., random slope of Time per subject)

### Step 5 — Run
Click **▶ Run Model**. Results appear in the main panel.

---

## Output

| Section | What you get |
|---------|-------------|
| **Model Fit** | AIC, BIC, Log-Likelihood, ICC |
| **Fixed Effects** | β estimates, SE, z-statistic, p-values, 95% CI, significance stars |
| **Random Effects** | Variance components, SD, ICC |
| **Plots** | Residuals vs Fitted, Q-Q plot, Scale-Location, Predicted vs Observed |
| **Export** | Full summary .txt, Fixed effects .csv |

---

## Sample Data
A sample dataset (`sample_data.csv`) is included with 20 participants, treatment/control groups, and pre/post sessions.

**Suggested settings for sample data:**
- Outcome: `outcome_score`
- Subject ID: `participant_id`
- Group: `group`
- Time: `session`
- Covariates: `age`, `baseline_score`
- Interactions: Group × Time ✓

---

## Technical Notes

- Uses `statsmodels.MixedLM` under the hood (equivalent to `lme4::lmer` in R)
- Random effects: REML-like fitting via `lbfgs` optimiser
- Significance stars: `***` p<0.001 · `**` p<0.01 · `*` p<0.05 · `.` p<0.1
- ICC = σ²_intercept / (σ²_intercept + σ²_residual)
