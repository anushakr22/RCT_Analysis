# RCT · Statistical Analysis Suite

A Streamlit web app for non-technical researchers to run statistical models on RCT data — no coding required.

Supports two models:
- **Linear Mixed Model (LMM)** — for continuous outcomes with repeated measures
- **Mixed Factorial ANOVA** — for factorial designs with between and within-subjects factors

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the app
streamlit run app.py
```

The app will open in your browser at http://localhost:8501

> **Note:** The app also requires a local `ingestion.py` file in the same directory. This handles file reading and data cleaning and is not a pip package.

---

## Requirements

```
streamlit>=1.32.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
statsmodels>=0.14.0
scipy>=1.10.0
openpyxl>=3.1.0
pingouin>=0.5.4
```

---

## How to Use

### Step 1 — Upload Data
Upload a **CSV** or **XLSX** file via the sidebar. Data should be in **long format**:
- One row per observation per participant per time point
- Columns for: participant ID, group, time/session, outcome, and any covariates

The app will automatically:
- Detect and remove junk/summary rows
- Handle merged cells (with a warning)
- Convert wide format to long format if needed
- Auto-detect which columns are the outcome, subject ID, group, and time variable

### Step 2 — Filter Data *(optional)*
Use the Filters panel to include/exclude specific groups, timepoints, or any categorical variable before modelling.

### Step 3 — Choose a Model
Select either **Linear Mixed Model** or **Mixed Factorial ANOVA** from the dropdown.

---

## Model 1 — Linear Mixed Model (LMM)

Best for: continuous outcomes measured repeatedly per participant.

### Variable Setup
| Role | Example column name |
|------|-------------------|
| Outcome | `depression_score`, `pain_rating` |
| Subject ID | `participant_id`, `id` |
| Group | `group` (treatment / control) |
| Time / Session | `session` (pre / post), `week` |

### Fixed Effects
- Group and Time are auto-added
- Add extra covariates (age, gender, baseline, etc.)
- Group × Time interaction offered automatically
- Custom interactions use `var1:var2` syntax

### Random Effects
- Random intercept per subject always included
- Optionally add random slopes (e.g. random slope of Time per subject)

### Output

| Section | What you get |
|---------|-------------|
| **Fixed Effects table** | β estimates, SE, z-statistic, p-values, 95% CI, significance stars |
| **Plain-language summary** | Interpretation of key findings in plain English |
| **Diagnostic plots** | Residuals vs Fitted, Q-Q, Scale-Location, Predicted vs Observed |
| **Export** | Full summary `.txt`, Fixed effects `.csv` |

---

## Model 2 — Mixed Factorial ANOVA

Best for: factorial designs where you want to test group differences, time effects, and their interaction.

### Variable Setup
| Role | Description |
|------|-------------|
| Outcome | Continuous dependent variable |
| Subject ID | Column identifying each participant |
| Between-subjects factor | Each person belongs to one level only (e.g. Group: Treatment vs Control) |
| Within-subjects factor | Every person appears at every level (e.g. Time: Pre, Post) |

Supports:
- Up to 2 between-subjects factors
- Up to 2 within-subjects factors
- Optional numeric covariates (turns it into an ANCOVA)

### Advanced Options
- **Post-hoc pairwise comparisons** — identifies exactly which pairs of groups or timepoints differ
- **Multiple comparison correction** — Bonferroni, Holm, FDR, or none
- **Sphericity test** (Mauchly's W) — checks ANOVA assumptions; Greenhouse-Geisser correction applied automatically if violated

### Output

| Section | What you get |
|---------|-------------|
| **ANOVA table** | F-statistic, p-value, partial η² (effect size), G-G correction |
| **Sphericity results** | Mauchly's W, p-value, verdict |
| **Post-hoc table** | Pairwise comparisons with corrected p-values |
| **Plain-language summary** | Interpretation of key findings in plain English |
| **Group × Time means plot** | Visual of how each group changed across timepoints |
| **Export** | ANOVA table `.csv`, Post-hoc results `.csv` |

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

**LMM**
- Uses `statsmodels.MixedLM` (equivalent to `lme4::lmer` in R)
- Fitting: Maximum Likelihood via `lbfgs` optimiser (`reml=False`)
- ICC = σ²_intercept / (σ²_intercept + σ²_residual)

**ANOVA**
- Uses `pingouin` for mixed ANOVA, repeated measures ANOVA, post-hoc tests, and sphericity
- Falls back to `statsmodels OLS + anova_lm` for between-subjects-only designs (Type II sums of squares)

**General**
- Significance stars: `***` p<0.001 · `**` p<0.01 · `*` p<0.05 · `.` p<0.1
- Effect size benchmarks: η² < 0.06 small · 0.06–0.14 medium · > 0.14 large
- Highlighted rows in all tables = significant at α = 0.05