"""
MISO Data Patch Script
Applies 4 fixes to MISO_Students_IT_Analysis_Ready.csv
Produces MISO_Students_IT_Analysis_Ready_v2.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the slim analysis-ready file
df = pd.read_csv("MISO_Students_IT_Analysis_Ready.csv", low_memory=False)
print(f"Loaded: {df.shape}")

# -----------------------------------------------------------------------
# PATCH 1 — Drop Exact Duplicate Rows
# -----------------------------------------------------------------------

before = len(df)
df = df.drop_duplicates().reset_index(drop=True)
after = len(df)
print(f"[PATCH 1] Dropped {before - after} duplicate rows. Remaining: {after}")

# -----------------------------------------------------------------------
# PATCH 2 — Fix HISP and INTER Binary Encoding
#
# Root cause: raw files use 0=No, 1=Yes — the original script mapped
# 1→True, 2→False (wrong), leaving all 0 values as NaN.
# The slim file has already lost the False values, so we re-extract
# HISP/INTER from the original source files using the same skip-row
# indices as the original pipeline to preserve row alignment.
# -----------------------------------------------------------------------

FILES = {
    2018: "2018 Student MISO Survey Results.csv",
    2021: "2021 Student MISO Survey Results.csv",
    2024: "2024 Student MISO Survey Results.csv",
}
# These match the original pipeline's skip_rows (same rows dropped → same alignment)
SKIP_ROWS = {2018: [1], 2021: [1, 2], 2024: [1, 2]}


def _extract_binary_col(year, col):
    """Re-read one column from a source file with correct 0/1 encoding."""
    raw = pd.read_csv(FILES[year], header=0, encoding='latin1', low_memory=False)
    raw = raw.drop(index=SKIP_ROWS[year]).reset_index(drop=True)
    if col not in raw.columns:
        return pd.Series([np.nan] * len(raw))
    s = pd.to_numeric(raw[col], errors='coerce')
    s = s.replace(-99, np.nan)             # sentinel → NaN
    s = s.map({0: False, 1: True})         # 0=No, 1=Yes
    return s


# Build per-year HISP/INTER, then concatenate in survey_year order
# The slim file is sorted: pre_covid (2018) → during_covid (2021) → post_covid (2024)
hisp_series = pd.concat(
    [_extract_binary_col(yr, 'HISP') for yr in [2018, 2021, 2024]],
    ignore_index=True
)
inter_series = pd.concat(
    [_extract_binary_col(yr, 'INTER') for yr in [2018, 2021, 2024]],
    ignore_index=True
)

# The original merged df had 830 rows; after Patch 1 we may have fewer.
# Identify which row indices survived dedup by tracking them before drop_duplicates.
# We need to align the re-read series to the surviving rows.
# Strategy: reload slim file (pre-dedup), find surviving row positions,
# then select those positions from the 830-length re-read series.
df_pre_dedup = pd.read_csv("MISO_Students_IT_Analysis_Ready.csv", low_memory=False)
pre_dedup_idx = df_pre_dedup.drop_duplicates().index.tolist()  # which original rows survived

# Select matching positions from the re-read 830-length HISP/INTER
hisp_aligned = hisp_series.iloc[pre_dedup_idx].values
inter_aligned = inter_series.iloc[pre_dedup_idx].values

# Assign into the already-deduped df (lengths must match)
assert len(hisp_aligned) == len(df), \
    f"Alignment error: hisp={len(hisp_aligned)}, df={len(df)}"

df['HISP'] = hisp_aligned
df['INTER'] = inter_aligned

for col in ['HISP', 'INTER']:
    valid = df[col].notna().sum()
    pct_valid = valid / len(df)
    n_true = (df[col] == True).sum()
    n_false = (df[col] == False).sum()
    print(f"[PATCH 2] {col}: {valid} valid ({pct_valid:.1%}) — "
          f"Yes={n_true}, No={n_false}, NaN={df[col].isna().sum()}")

# -----------------------------------------------------------------------
# PATCH 3 — Fix Composite Scores (Remove Out-of-Range Values)
# -----------------------------------------------------------------------

# Step 3a — Clamp raw Likert columns to valid ranges
USE_cols  = [c for c in df.columns if c.startswith('USE_')  and not c.endswith('_NA')]
IMP_cols  = [c for c in df.columns if c.startswith('IMP_')  and not c.endswith('_NA')]
DS_cols   = [c for c in df.columns if c.startswith('DS_')   and not c.endswith('_NA')]
INF_cols  = [c for c in df.columns if c.startswith('INF_')  and not c.endswith('_NA')]
DAHD_cols = [c for c in df.columns if c.startswith('DAHD_') and not c.endswith('_NA')]
AAG_cols  = [c for c in df.columns if c.startswith('AAG_')  and not c.endswith('_NA')]


def clamp(df, cols, lo, hi):
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].where(df[col].between(lo, hi), np.nan)
    return df


df = clamp(df, USE_cols + IMP_cols + DS_cols, 1, 5)
df = clamp(df, INF_cols + DAHD_cols + AAG_cols, 1, 4)

print(f"\n[PATCH 3a] Clamped Likert columns to valid ranges.")
if USE_cols + IMP_cols + DS_cols:
    print(f"  USE/IMP/DS max after clamp: "
          f"{df[USE_cols + IMP_cols + DS_cols].max().max():.1f} (expect <= 5)")
if INF_cols + DAHD_cols + AAG_cols:
    print(f"  INF/DAHD/AAG max after clamp: "
          f"{df[INF_cols + DAHD_cols + AAG_cols].max().max():.1f} (expect <= 4)")

# Step 3b — Recompute composites from cross-year core columns only
# (columns present in all 3 years → directly comparable across COVID periods)
CORE_USE  = ['USE_CMS', 'USE_FPC', 'USE_AORO', 'USE_ERPSS', 'USE_CWS']
CORE_IMP  = ['IMP_CMS', 'IMP_FPC', 'IMP_SICP', 'IMP_SDCP', 'IMP_AWAC',
             'IMP_PWAC', 'IMP_AORO', 'IMP_ERPSS', 'IMP_SERPP', 'IMP_CWS', 'IMP_OCS']
CORE_DS   = ['DS_CMS', 'DS_CMSS', 'DS_FPC', 'DS_SICP', 'DS_SDCP',
             'DS_AWAC', 'DS_PWAC', 'DS_AORO', 'DS_ERPSS', 'DS_SERPP', 'DS_CWS', 'DS_OCS']
CORE_DAHD = ['DAHD_F', 'DAHD_K', 'DAHD_RL', 'DAHD_RS']
CORE_DIG  = ['USE_CMS', 'USE_AORO', 'USE_CWS']
CORE_INF  = ['INF_ATS']


def composite(df, cols, name, na_threshold=0.5):
    """Mean of present cols; NaN if >na_threshold fraction are missing."""
    present = [c for c in cols if c in df.columns]
    if not present:
        df[name] = np.nan
        return df
    df[name] = df[present].mean(axis=1, skipna=True)
    na_frac = df[present].isna().mean(axis=1)
    df.loc[na_frac > na_threshold, name] = np.nan
    return df


df = composite(df, CORE_USE,  'composite_IT_usage')
df = composite(df, CORE_IMP,  'composite_IT_importance')
df = composite(df, CORE_DS,   'composite_IT_satisfaction')
df = composite(df, CORE_DAHD, 'composite_helpdesk_staff')
df = composite(df, CORE_DIG,  'digital_engagement_index')
df = composite(df, CORE_INF,  'composite_IT_awareness')

print(f"\n[PATCH 3b] Recomputed composites (cross-year core columns only).")
print(df.groupby('covid_period')[
    ['composite_IT_usage', 'composite_IT_importance',
     'composite_IT_satisfaction', 'digital_engagement_index']
].mean().round(3))

comp_max = df[['composite_IT_usage', 'composite_IT_importance',
               'composite_IT_satisfaction']].max().max()
print(f"  All composite maxes <= 5: {comp_max:.2f}")

# -----------------------------------------------------------------------
# PATCH 4 — Fix the Importance-Satisfaction Gap Direction
# -----------------------------------------------------------------------

# Gap: positive = students care more than they are satisfied (service gap)
df['IT_importance_satisfaction_gap'] = (
    df['composite_IT_importance'] - df['composite_IT_satisfaction']
)

# satisfaction_deficit: same as gap but floored at 0 (for visualizations)
df['satisfaction_deficit'] = df['IT_importance_satisfaction_gap'].apply(
    lambda x: max(x, 0) if pd.notna(x) else np.nan
)

print(f"\n[PATCH 4] Gap stats by covid_period:")
print(df.groupby('covid_period')[
    ['IT_importance_satisfaction_gap', 'satisfaction_deficit']
].mean().round(3))
print(
    "Interpretation guide:\n"
    "  IT_importance_satisfaction_gap > 0  = students care more than they are satisfied (service gap)\n"
    "  IT_importance_satisfaction_gap < 0  = students are more satisfied than they rate importance\n"
    "  satisfaction_deficit                = same as gap but floored at 0 (use for viz)"
)

# -----------------------------------------------------------------------
# FINAL VALIDATION
# -----------------------------------------------------------------------

print("=" * 60)
print("PATCH VALIDATION")
print("=" * 60)

assert df.duplicated().sum() == 0, "FAIL: Still has duplicates"
print("[OK] No duplicate rows")

hisp_valid = df['HISP'].notna().sum()
inter_valid = df['INTER'].notna().sum()
assert hisp_valid > 50, f"FAIL: HISP still mostly missing: {hisp_valid}"
assert inter_valid > 50, f"FAIL: INTER still mostly missing: {inter_valid}"
print(f"[OK] HISP: {hisp_valid} valid | INTER: {inter_valid} valid")

max_composite = df[['composite_IT_usage', 'composite_IT_importance',
                     'composite_IT_satisfaction']].max().max()
assert max_composite <= 5.0, f"FAIL: Composite scores exceed 5.0: {max_composite}"
print(f"[OK] All composites within 1-5 range (max = {max_composite:.2f})")

gap_mean = df.groupby('covid_period')['IT_importance_satisfaction_gap'].mean()
print(f"[OK] Gap by period:\n{gap_mean.round(3)}")

print(f"\nFinal shape: {df.shape}")
print(f"Rows per period:\n{df.groupby('covid_period').size()}")

# -----------------------------------------------------------------------
# EXPORT
# -----------------------------------------------------------------------

df.to_csv("MISO_Students_IT_Analysis_Ready_v2.csv", index=False)
print("\n[OK] Saved: MISO_Students_IT_Analysis_Ready_v2.csv")
print(f"   Shape: {df.shape}")

# Regenerate diagnostic chart with corrected composites
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
composites_to_plot = ['composite_IT_usage', 'composite_IT_satisfaction',
                      'digital_engagement_index']
titles = ['IT Service Usage', 'IT Satisfaction', 'Digital Engagement Index']
colors = ['#4C72B0', '#C44E52', '#55A868']
order = ['pre_covid', 'during_covid', 'post_covid']
labels = ['Pre-COVID\n(2018)', 'During-COVID\n(2021)', 'Post-COVID\n(2024)']

for ax, col, title in zip(axes, composites_to_plot, titles):
    vals = df.groupby('covid_period')[col].mean().reindex(order)
    bars = ax.bar(labels, vals, color=colors)
    for bar, val in zip(bars, vals):
        if not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.03,
                    f'{val:.2f}', ha='center', va='bottom',
                    fontsize=11, fontweight='bold')
    ax.set_title(f'Mean {title}\nby COVID Period', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Score (1–5)')
    ax.set_ylim(1, 5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.suptitle('MISO Student Survey — IT Services Overview (Patched)',
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig("MISO_Diagnostic_Overview_v2.png", dpi=150, bbox_inches='tight')
plt.close()
print("[OK] Saved: MISO_Diagnostic_Overview_v2.png")
