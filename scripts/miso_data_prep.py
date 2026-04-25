"""
MISO Student Survey Data Preparation
Pre-COVID (2018) / During-COVID (2021) / Post-COVID (2024)
IT Services Only — Framingham State University
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
import json
warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# STEP 1: ENVIRONMENT SETUP
# ---------------------------------------------------------------------------

FILES = {
    2018: "2018 Student MISO Survey Results.csv",
    2021: "2021 Student MISO Survey Results.csv",
    2024: "2024 Student MISO Survey Results.csv",
}
# After reading with header=0, these are the 0-indexed data rows to drop
# (they are label/metadata rows, not real respondent data)
SKIP_ROWS = {2018: [1], 2021: [1, 2], 2024: [1, 2]}

COVID_PERIOD = {2018: 'pre_covid', 2021: 'during_covid', 2024: 'post_covid'}

# ---------------------------------------------------------------------------
# STEP 2: LOAD AND CLEAN HEADERS
# ---------------------------------------------------------------------------

def load_year(year):
    """Load one survey year, drop metadata rows, tag with year/period."""
    df = pd.read_csv(FILES[year], header=0, encoding='latin1', low_memory=False)
    # Drop metadata/label rows by integer position (0-indexed after header)
    df = df.drop(index=SKIP_ROWS[year]).reset_index(drop=True)
    df['survey_year'] = year
    df['covid_period'] = COVID_PERIOD[year]
    return df

print("Loading files...")
raw = {}
for yr in [2018, 2021, 2024]:
    raw[yr] = load_year(yr)
    print(f"  {yr}: {raw[yr].shape[0]} rows, {raw[yr].shape[1]} cols")

# ---------------------------------------------------------------------------
# STEP 3: FILTER TO IT-ONLY COLUMNS
# ---------------------------------------------------------------------------

# Library-related substrings to EXCLUDE
LIB_SUBSTRINGS = [
    '_LRS', '_LIAC', '_OLC', '_LSG', '_LC', '_LEC', '_ILL', '_LDB', '_DIC',
    '_PCL', '_LCS', '_PCR', '_QWSL', '_GSSL', '_PCIL', '_FPML', '_LWS',
    '_PDLC', '_LS', '_WL', 'INF_ALS', 'AAG_PDLC', 'AAG_LS', 'AAG_WL',
    'LRN_LDB', 'USE_LDL', 'IMP_LDL', 'DS_LDL', 'USE_LVSS', 'IMP_LVSS',
    'DS_LVSS', 'USE_ASC', 'IMP_ASC', 'DS_ASC', 'USE_LCS', 'USE_TLC',
    'IMP_TLC', 'DS_TLC', 'EITS', 'ELS', 'REM_SDCP',
    # DALC and DALR are library staff rating columns
    'DALC_', 'DALR_',
]

# Columns to always drop
DROP_COLS = [
    'StartDate', 'EndDate', 'Finished', 'DistributionChannel', 'UserLanguage',
    'COMMENT', 'V8', 'V9', 'V10',
]

# Demographic / metadata columns to always keep
KEEP_ALWAYS = {
    'YEAR', 'ADIV', 'AGE', 'SEX', 'SEX_TEXT', 'SEX_18_TEXT',
    'RACE5_1', 'RACE5_2', 'RACE5_3', 'RACE5_4', 'RACE5_5', 'RACE5_6',
    'HISP', 'INTER', 'NOTFGEN',
    'survey_year', 'covid_period',
}


def is_library_col(col):
    """Return True if column should be excluded as a library column."""
    for sub in LIB_SUBSTRINGS:
        if sub in col:
            return True
    return False


def filter_it_cols(df, year):
    """Keep IT + demographic columns; drop library and admin columns."""
    kept = []
    for col in df.columns:
        if col in DROP_COLS:
            continue
        if col.startswith('Unnamed'):
            continue
        if col in KEEP_ALWAYS:
            kept.append(col)
            continue
        if is_library_col(col):
            continue
        kept.append(col)
    return df[kept].copy()


filtered = {}
for yr in [2018, 2021, 2024]:
    filtered[yr] = filter_it_cols(raw[yr], yr)
    print(f"  {yr} after IT filter: {filtered[yr].shape[1]} cols")

# ---------------------------------------------------------------------------
# STEP 4: HANDLE SENTINEL VALUES AND MISSING DATA
# ---------------------------------------------------------------------------

# Identify Likert columns (USE_, IMP_, DS_, INF_, DAHD_, AAG_)
LIKERT_PREFIXES = ('USE_', 'IMP_', 'DS_', 'INF_', 'DAHD_', 'AAG_')
DEMO_COLS = {'YEAR', 'ADIV', 'AGE', 'SEX', 'SEX_TEXT', 'SEX_18_TEXT',
             'HISP', 'INTER', 'NOTFGEN',
             'RACE5_1', 'RACE5_2', 'RACE5_3', 'RACE5_4', 'RACE5_5', 'RACE5_6'}


def get_likert_cols(df):
    return [c for c in df.columns if c.startswith(LIKERT_PREFIXES)]


def handle_sentinels(df, year):
    """
    Replace -99 (sentinel) with NaN.
    Track NA flags for Likert cols with >10% -99 rate.
    Impute Likert cols <40% missing with median; flag >=40%.
    """
    df = df.copy()
    likert_cols = get_likert_cols(df)
    high_missing = []

    # Force numeric conversion on Likert cols (some may be strings)
    for col in likert_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Replace -99 with NaN
    for col in likert_cols:
        na_flag = df[col] == -99
        df[col] = df[col].where(~na_flag, np.nan)

        # Track NA flag if rate > 10%
        na_rate = na_flag.mean()
        if na_rate > 0.10:
            df[f'{col}_NA'] = na_flag

    # Missing data summary
    n = len(df)
    summary_rows = []
    for col in likert_cols:
        missing_n = df[col].isna().sum()
        missing_pct = missing_n / n
        summary_rows.append({'col': col, 'missing_n': missing_n, 'missing_pct': missing_pct})

    summary_df = pd.DataFrame(summary_rows).sort_values('missing_pct', ascending=False)
    print(f"\n  {year} — Top 10 missing Likert cols:")
    print(summary_df.head(10).to_string(index=False))

    # Impute or flag
    for _, row in summary_df.iterrows():
        col = row['col']
        if row['missing_pct'] >= 0.40:
            high_missing.append(col)
        else:
            med = df[col].median()
            df[col] = df[col].fillna(med)

    if high_missing:
        print(f"  {year} HIGH-MISSING cols (>=40%): {high_missing}")

    # Demo cols: replace -99 with NaN, no imputation
    for col in df.columns:
        if col in DEMO_COLS:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].replace(-99, np.nan)

    return df, high_missing


cleaned = {}
high_missing_all = {}
for yr in [2018, 2021, 2024]:
    print(f"\nProcessing sentinels for {yr}...")
    cleaned[yr], high_missing_all[yr] = handle_sentinels(filtered[yr], yr)

# ---------------------------------------------------------------------------
# STEP 5: ENCODE LIKERT SCALES TO NUMERIC
# ---------------------------------------------------------------------------

# Usage frequency (USE_*)
USE_MAP = {
    'Never': 1,
    'Once or twice a semester': 2,
    'One to three times a month': 3,
    'One to three times a week': 4,
    'More than three times a week': 5,
}

# Importance (IMP_*)
IMP_MAP = {
    'Not important': 1,
    'Somewhat important': 2,
    'Important': 3,
    'Very important': 4,
    'Not applicable': np.nan,
}

# Satisfaction (DS_*)
DS_MAP = {
    'Dissatisfied': 1,
    'Somewhat dissatisfied': 2,
    'Somewhat satisfied': 3,
    'Satisfied': 4,
    'Not applicable': np.nan,
}

# Informed (INF_*)
INF_MAP = {
    'Not informed': 1,
    'Somewhat informed': 2,
    'Informed': 3,
    'Very informed': 4,
}

# Agree/Disagree (DAHD_*)
DAHD_MAP = {
    'Disagree': 1,
    'Somewhat disagree': 2,
    'Somewhat agree': 3,
    'Agree': 4,
    'Not applicable': np.nan,
}

# Academic goals (AAG_*)
AAG_MAP = {
    'Not at all': 1,
    'Slightly': 2,
    'Moderately': 3,
    'Greatly': 4,
    'Not applicable': np.nan,
}


def encode_likert(df):
    """Apply text-to-int mappings if string labels remain; cast all to float."""
    df = df.copy()
    for col in df.columns:
        if col.startswith('USE_'):
            df[col] = df[col].replace(USE_MAP)
        elif col.startswith('IMP_'):
            df[col] = df[col].replace(IMP_MAP)
        elif col.startswith('DS_'):
            df[col] = df[col].replace(DS_MAP)
        elif col.startswith('INF_'):
            df[col] = df[col].replace(INF_MAP)
        elif col.startswith('DAHD_'):
            df[col] = df[col].replace(DAHD_MAP)
        elif col.startswith('AAG_'):
            df[col] = df[col].replace(AAG_MAP)
        else:
            continue
        df[col] = pd.to_numeric(df[col], errors='coerce').astype(float)
    return df


for yr in [2018, 2021, 2024]:
    cleaned[yr] = encode_likert(cleaned[yr])
print("\nLikert encoding complete.")

# ---------------------------------------------------------------------------
# STEP 6: STANDARDIZE DEMOGRAPHIC COLUMNS
# ---------------------------------------------------------------------------

ADIV_MAP_2018 = {
    45700: 'Arts_Humanities',
    52200: 'Social_Behavioral_Sciences',
    20100: 'STEM',
    60200: 'Education',
    '45700': 'Arts_Humanities',
    '52200': 'Social_Behavioral_Sciences',
    '20100': 'STEM',
    '60200': 'Education',
}

ADIV_MAP_2021_2024 = {
    119903: 'Arts_Humanities',
    119902: 'Business',
    70100: 'Education_Social_Behavioral',
    20100: 'STEM',
    115100: 'Graduate_Continuing_Ed',
    '119903': 'Arts_Humanities',
    '119902': 'Business',
    '70100': 'Education_Social_Behavioral',
    '20100': 'STEM',
    '115100': 'Graduate_Continuing_Ed',
}

AGE_MAP = {
    1: '18_or_younger', 2: '19', 3: '20', 4: '21', 5: '22', 6: '23_or_older',
    7: '23_or_older',  # catch-all for any 7-category version
}

GENDER_MAP = {
    1: 'Female', 2: 'Male', 3: 'Non_binary',
    4: 'Self_describe', 5: 'Prefer_not_say',
}

RACE_RENAME = {
    'RACE5_1': 'race_white',
    'RACE5_2': 'race_black_african_american',
    'RACE5_3': 'race_asian',
    'RACE5_4': 'race_american_indian_alaska_native',
    'RACE5_5': 'race_native_hawaiian_pacific_islander',
    'RACE5_6': 'race_other',
}


def standardize_demographics(df, year):
    df = df.copy()

    # ADIV
    adiv_map = ADIV_MAP_2018 if year == 2018 else ADIV_MAP_2021_2024
    if 'ADIV' in df.columns:
        df['ADIV'] = pd.to_numeric(df['ADIV'], errors='coerce')
        df['adiv_label'] = df['ADIV'].map(adiv_map).fillna('Other/Unknown')

    # Gender — unify SEX / SEX_18_TEXT into single 'gender' column
    sex_col = 'SEX' if 'SEX' in df.columns else None
    if sex_col:
        df[sex_col] = pd.to_numeric(df[sex_col], errors='coerce')
        df['gender'] = df[sex_col].map(GENDER_MAP).fillna('Not_reported')

    # AGE
    if 'AGE' in df.columns:
        df['AGE'] = pd.to_numeric(df['AGE'], errors='coerce')
        df['age_label'] = df['AGE'].map(AGE_MAP).fillna('Not_reported')

    # HISP / INTER — binary
    for col in ['HISP', 'INTER']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].map({1: True, 2: False})

    # NOTFGEN (2024 only)
    if 'NOTFGEN' in df.columns:
        df['NOTFGEN'] = pd.to_numeric(df['NOTFGEN'], errors='coerce')
        df['NOTFGEN'] = df['NOTFGEN'].map({1: True, 2: False})

    # RACE5 — rename and ensure binary
    rename_present = {k: v for k, v in RACE_RENAME.items() if k in df.columns}
    df = df.rename(columns=rename_present)
    for new_col in rename_present.values():
        df[new_col] = pd.to_numeric(df[new_col], errors='coerce')

    # YEAR (graduation year) — numeric; create years_to_grad
    if 'YEAR' in df.columns:
        df['YEAR'] = pd.to_numeric(df['YEAR'], errors='coerce')
        df['YEAR'] = df['YEAR'].where(df['YEAR'] > 0)
        df['years_to_grad'] = df['YEAR'] - year

    return df


for yr in [2018, 2021, 2024]:
    cleaned[yr] = standardize_demographics(cleaned[yr], yr)
print("Demographics standardized.")

# ---------------------------------------------------------------------------
# STEP 7: HARMONIZE COLUMN NAMES — TRACK COVERAGE
# ---------------------------------------------------------------------------

COLUMN_COVERAGE = {}
for yr in [2018, 2021, 2024]:
    for col in cleaned[yr].columns:
        if col not in COLUMN_COVERAGE:
            COLUMN_COVERAGE[col] = []
        COLUMN_COVERAGE[col].append(yr)

# Note year-specific columns
year_only = {col: yrs for col, yrs in COLUMN_COVERAGE.items() if len(yrs) == 1}
print(f"\nYear-specific columns: {len(year_only)}")
for col, yrs in sorted(year_only.items()):
    print(f"  {col}: {yrs}")

# ---------------------------------------------------------------------------
# STEP 8: FEATURE ENGINEERING
# ---------------------------------------------------------------------------

def engineer_features(df):
    df = df.copy()

    it_cols = set(df.columns)

    # IT Usage composite
    use_cols = [c for c in df.columns if c.startswith('USE_') and not c.endswith('_NA')]
    df['composite_IT_usage'] = df[use_cols].mean(axis=1, skipna=True)

    # IT Importance composite
    imp_cols = [c for c in df.columns if c.startswith('IMP_') and not c.endswith('_NA')]
    df['composite_IT_importance'] = df[imp_cols].mean(axis=1, skipna=True)

    # IT Satisfaction composite
    ds_cols = [c for c in df.columns if c.startswith('DS_') and not c.endswith('_NA')]
    df['composite_IT_satisfaction'] = df[ds_cols].mean(axis=1, skipna=True)

    # Helpdesk staff quality composite
    dahd_cols = [c for c in ['DAHD_F', 'DAHD_K', 'DAHD_RL', 'DAHD_RS'] if c in df.columns]
    df['composite_helpdesk_staff'] = df[dahd_cols].mean(axis=1, skipna=True) if dahd_cols else np.nan

    # Awareness composite
    inf_cols = [c for c in ['INF_ATS'] if c in df.columns]
    df['composite_IT_awareness'] = df[inf_cols].mean(axis=1, skipna=True) if inf_cols else np.nan

    # Satisfaction-Importance gap (positive = important but not satisfied)
    df['IT_importance_satisfaction_gap'] = df['composite_IT_importance'] - df['composite_IT_satisfaction']

    # Digital engagement index (remote/digital services — critical for COVID comparison)
    # USE_CMS = LMS (Blackboard/Canvas), USE_AORO = off-campus access, USE_CWS = ITS website
    digital_cols = [c for c in ['USE_CMS', 'USE_AORO', 'USE_CWS'] if c in df.columns]
    df['digital_engagement_index'] = df[digital_cols].mean(axis=1, skipna=True) if digital_cols else np.nan

    # Device ownership count
    own_cols = [c for c in df.columns if c.startswith('OWN_')]
    if own_cols:
        df['device_ownership_count'] = df[own_cols].apply(
            lambda row: pd.to_numeric(row, errors='coerce').eq(1).sum(), axis=1
        )
    else:
        df['device_ownership_count'] = np.nan

    return df


for yr in [2018, 2021, 2024]:
    cleaned[yr] = engineer_features(cleaned[yr])
print("Feature engineering complete.")

# ---------------------------------------------------------------------------
# STEP 9: OUTLIER DETECTION
# ---------------------------------------------------------------------------

COMPOSITE_COLS = [
    'composite_IT_usage', 'composite_IT_importance', 'composite_IT_satisfaction',
    'composite_helpdesk_staff', 'digital_engagement_index',
]


def flag_outliers(df):
    df = df.copy()
    outlier_mask = pd.Series(False, index=df.index)
    for col in COMPOSITE_COLS:
        if col not in df.columns:
            continue
        vals = pd.to_numeric(df[col], errors='coerce')
        q1, q3 = vals.quantile(0.25), vals.quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outlier_mask |= (vals < lo) | (vals > hi)
    df['is_outlier_composite'] = outlier_mask
    return df


for yr in [2018, 2021, 2024]:
    cleaned[yr] = flag_outliers(cleaned[yr])
    n_out = cleaned[yr]['is_outlier_composite'].sum()
    print(f"  {yr}: {n_out} outlier respondents flagged ({n_out/len(cleaned[yr]):.1%})")

# ---------------------------------------------------------------------------
# STEP 10: MERGE ALL THREE YEARS
# ---------------------------------------------------------------------------

from pandas.api.types import CategoricalDtype

merged_df = pd.concat(
    [cleaned[2018], cleaned[2021], cleaned[2024]],
    axis=0, ignore_index=True, sort=False
)

period_order = CategoricalDtype(['pre_covid', 'during_covid', 'post_covid'], ordered=True)
merged_df['covid_period'] = merged_df['covid_period'].astype(period_order)
merged_df = merged_df.sort_values('covid_period').reset_index(drop=True)

print(f"\nFinal merged shape: {merged_df.shape}")
print(f"Rows per period:\n{merged_df['covid_period'].value_counts().sort_index()}")

# ---------------------------------------------------------------------------
# STEP 11: DATA QUALITY VALIDATION
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("DATA QUALITY VALIDATION REPORT")
print("=" * 60)

print(f"\n[1] Shape: {merged_df.shape}")
print(merged_df.groupby(['survey_year', 'covid_period'], observed=True).size().rename('n_respondents'))

dupes = merged_df.duplicated().sum()
print(f"\n[2] Duplicate rows: {dupes}")

it_likert = [c for c in merged_df.columns
             if c.startswith(('USE_', 'IMP_', 'DS_', 'INF_', 'DAHD_', 'AAG_'))
             and not c.endswith('_NA')]
missing_pct = merged_df[it_likert].isna().mean().sort_values(ascending=False)
print(f"\n[3] Top 10 columns by missing %:")
print(missing_pct.head(10).map('{:.1%}'.format))

print(f"\n[4] Value range check on Likert columns (sample):")
for col in it_likert[:5]:
    vals = pd.to_numeric(merged_df[col], errors='coerce').dropna()
    if len(vals):
        print(f"  {col}: min={vals.min():.0f}, max={vals.max():.0f}")

composites = ['composite_IT_usage', 'composite_IT_importance', 'composite_IT_satisfaction',
              'composite_helpdesk_staff', 'digital_engagement_index']
print(f"\n[5] Composite score summary by covid_period:")
print(merged_df.groupby('covid_period', observed=True)[composites].mean().round(3))

print(f"\n[6] Gender distribution:")
print(merged_df.groupby(['survey_year', 'gender']).size().unstack(fill_value=0))

high_missing_overall = missing_pct[missing_pct > 0.40].index.tolist()
print(f"\n[7] High-missing columns (>40%):")
print(high_missing_overall if high_missing_overall else "None")

print("=" * 60)

# ---------------------------------------------------------------------------
# STEP 12: EXPORT OUTPUTS
# ---------------------------------------------------------------------------

# 1. Full merged dataset
merged_df.to_csv("MISO_Students_Cleaned_2018_2021_2024.csv", index=False)
print("\nSaved: MISO_Students_Cleaned_2018_2021_2024.csv")

# 2. Slim IT-only analysis-ready dataset
slim_cols = (
    ['survey_year', 'covid_period'] +
    [c for c in merged_df.columns
     if c.startswith(('USE_', 'IMP_', 'DS_', 'INF_', 'DAHD_', 'AAG_'))
     and not c.endswith('_NA')] +
    ['composite_IT_usage', 'composite_IT_importance', 'composite_IT_satisfaction',
     'composite_helpdesk_staff', 'composite_IT_awareness', 'digital_engagement_index',
     'IT_importance_satisfaction_gap', 'device_ownership_count',
     'gender', 'age_label', 'adiv_label', 'HISP', 'INTER', 'is_outlier_composite']
)
slim_cols_present = [c for c in slim_cols if c in merged_df.columns]
# deduplicate preserving order
seen = set()
slim_cols_dedup = [c for c in slim_cols_present if not (c in seen or seen.add(c))]
merged_df[slim_cols_dedup].to_csv("MISO_Students_IT_Analysis_Ready.csv", index=False)
print("Saved: MISO_Students_IT_Analysis_Ready.csv")

# 3. Column coverage JSON (convert int keys to str for JSON serialization)
coverage_serializable = {col: yrs for col, yrs in COLUMN_COVERAGE.items()}
with open("MISO_Column_Coverage.json", "w") as f:
    json.dump(coverage_serializable, f, indent=2)
print("Saved: MISO_Column_Coverage.json")

# 4. Data quality report as text
report_lines = [
    "=" * 60,
    "DATA QUALITY VALIDATION REPORT",
    "=" * 60,
    f"Shape: {merged_df.shape}",
    "",
    "Rows per period:",
    str(merged_df.groupby(['survey_year', 'covid_period'], observed=True).size().rename('n_respondents')),
    f"\nDuplicate rows: {dupes}",
    "\nTop 10 missing Likert cols:",
    str(missing_pct.head(10).map('{:.1%}'.format)),
    "\nComposite scores by covid_period:",
    str(merged_df.groupby('covid_period', observed=True)[composites].mean().round(3)),
    f"\nHigh-missing cols (>40%): {high_missing_overall if high_missing_overall else 'None'}",
    "=" * 60,
]
with open("MISO_Data_Quality_Report.txt", "w") as f:
    f.write("\n".join(report_lines))
print("Saved: MISO_Data_Quality_Report.txt")

print("\nAll outputs saved successfully.")
print("Files created:")
print("  - MISO_Students_Cleaned_2018_2021_2024.csv  (full merged)")
print("  - MISO_Students_IT_Analysis_Ready.csv       (slim IT-only)")
print("  - MISO_Column_Coverage.json                 (column availability by year)")
print("  - MISO_Data_Quality_Report.txt              (validation report)")

# ---------------------------------------------------------------------------
# STEP 13: DIAGNOSTIC VISUALIZATION
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
composites_to_plot = ['composite_IT_usage', 'composite_IT_satisfaction', 'digital_engagement_index']
titles = ['IT Service Usage', 'IT Satisfaction', 'Digital Engagement Index']
colors = ['steelblue', 'firebrick', 'seagreen']

period_labels = ['Pre-COVID\n(2018)', 'During-COVID\n(2021)', 'Post-COVID\n(2024)']

for ax, col, title in zip(axes, composites_to_plot, titles):
    if col not in merged_df.columns:
        ax.set_title(f'{title}\n(no data)')
        continue
    means = merged_df.groupby('covid_period', observed=True)[col].mean()
    # Reindex to ensure correct order
    means = means.reindex(['pre_covid', 'during_covid', 'post_covid'])
    bars = ax.bar(range(len(means)), means.values, color=colors[:len(means)])
    ax.set_title(f'Mean {title}\nby COVID Period')
    ax.set_ylabel('Mean Score (1–5)')
    ax.set_xticks(range(len(means)))
    ax.set_xticklabels(period_labels[:len(means)], rotation=0, fontsize=8)
    ax.set_ylim(1, 5)
    # Add value labels
    for bar, val in zip(bars, means.values):
        if not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig("MISO_Diagnostic_Overview.png", dpi=150, bbox_inches='tight')
plt.close()
print("Diagnostic plot saved: MISO_Diagnostic_Overview.png")
