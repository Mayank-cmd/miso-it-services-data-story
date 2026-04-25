
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from scipy import stats
import warnings
import os

warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════
# 1. SETUP & CONFIGURATION
# ══════════════════════════════════════════════════════════════

# Load dataset
print("=" * 70)
print("MISO STUDENT SURVEY — EXPLORATORY DATA ANALYSIS")
print("=" * 70)

CSV_FILE = "MISO_Students_IT_Analysis_Ready_v2.csv"

try:
    df = pd.read_csv(CSV_FILE, low_memory=False)
    print(f"\nLoaded: {CSV_FILE}")
    print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
except FileNotFoundError:
    print(f"\nERROR: Could not find '{CSV_FILE}'.")
    print("Please place the file in the same directory as this script.")
    exit(1)

# Output directory for figures
OUTPUT_DIR = "eda_figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Figures will be saved to: {os.path.abspath(OUTPUT_DIR)}/\n")

# ── Color palette ──
BLUE = '#2D5F8A'
RED = '#C0392B'
GREEN = '#27874A'
AMBER = '#D4920B'
GREY = '#6B7280'

PERIOD_COLORS = {
    'pre_covid': BLUE,
    'during_covid': RED,
    'post_covid': GREEN
}
PERIOD_LABELS = {
    'pre_covid': 'Pre-COVID\n(2018)',
    'during_covid': 'During-COVID\n(2021)',
    'post_covid': 'Post-COVID\n(2024)'
}
PERIOD_LABELS_INLINE = {
    'pre_covid': 'Pre-COVID (2018)',
    'during_covid': 'During-COVID (2021)',
    'post_covid': 'Post-COVID (2024)'
}
ORDER = ['pre_covid', 'during_covid', 'post_covid']

# ── Matplotlib defaults ──
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 10,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.grid': False,
})

# ── Service name lookups ──
CORE_DS = [
    'DS_CMS', 'DS_CMSS', 'DS_FPC', 'DS_SICP', 'DS_SDCP',
    'DS_AWAC', 'DS_PWAC', 'DS_AORO', 'DS_ERPSS', 'DS_SERPP',
    'DS_CWS', 'DS_OCS'
]
DS_NAMES = {
    'DS_CMS': 'Canvas/LMS', 'DS_CMSS': 'LMS Support',
    'DS_FPC': 'Help Desk', 'DS_SICP': 'Computing Status',
    'DS_SDCP': 'Desktop Support', 'DS_AWAC': 'WiFi Availability',
    'DS_PWAC': 'WiFi Performance', 'DS_AORO': 'Off-Campus Access',
    'DS_ERPSS': 'Banner/MyFram.', 'DS_SERPP': 'Banner Support',
    'DS_CWS': 'ITS Website', 'DS_OCS': 'Overall Computing'
}
CORE_USE = ['USE_CMS', 'USE_FPC', 'USE_AORO', 'USE_ERPSS', 'USE_CWS']
USE_NAMES = {
    'USE_CMS': 'Canvas/LMS', 'USE_FPC': 'Help Desk',
    'USE_AORO': 'Off-Campus Access', 'USE_ERPSS': 'Banner/MyFram.',
    'USE_CWS': 'ITS Website'
}
USE_FREQ_LABELS = {
    1: 'Never', 2: '1-2x/sem', 3: '1-3x/mo',
    4: '1-3x/wk', 5: '3+x/wk'
}

COMPOSITES = [
    'composite_IT_usage', 'composite_IT_importance',
    'composite_IT_satisfaction', 'digital_engagement_index',
    'composite_IT_awareness', 'composite_helpdesk_staff'
]
COMPOSITE_LABELS = {
    'composite_IT_usage': 'IT Usage (1–5)',
    'composite_IT_importance': 'IT Importance (1–4)',
    'composite_IT_satisfaction': 'IT Satisfaction (1–4)',
    'digital_engagement_index': 'Digital Engagement (1–5)',
    'composite_IT_awareness': 'IT Awareness (1–4)',
    'composite_helpdesk_staff': 'Helpdesk Staff (1–4)',
}

IMP_SAT_PAIRS = [
    ('IMP_PWAC', 'DS_PWAC', 'WiFi Perf.'),
    ('IMP_AWAC', 'DS_AWAC', 'WiFi Avail.'),
    ('IMP_ERPSS', 'DS_ERPSS', 'Banner'),
    ('IMP_CMS', 'DS_CMS', 'Canvas'),
    ('IMP_AORO', 'DS_AORO', 'Off-Campus'),
    ('IMP_FPC', 'DS_FPC', 'Help Desk'),
    ('IMP_OCS', 'DS_OCS', 'Overall IT'),
    ('IMP_SICP', 'DS_SICP', 'Comp. Status'),
]

DAHD_NAMES = {
    'DAHD_F': 'Friendly', 'DAHD_K': 'Knowledgeable',
    'DAHD_RL': 'Responsive', 'DAHD_RS': 'Resolved Issue'
}


def savefig(fig, filename):
    """Save figure to output directory and display it."""
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.show()  # Display in Spyder's Plots pane
    plt.close(fig)
    print(f"  Saved: {filename}")


# ══════════════════════════════════════════════════════════════
# 2. STATISTICAL ANALYSIS (Console Output)
# ══════════════════════════════════════════════════════════════

print("=" * 70)
print("STATISTICAL RESULTS")
print("=" * 70)

# ── 2.1 Composite score means by period ──
print("\n--- 2.1 Composite Score Means by Period ---")
for comp in COMPOSITES + ['IT_importance_satisfaction_gap']:
    print(f"\n  {comp}:")
    for period in ORDER:
        subset = df[df['covid_period'] == period][comp].dropna()
        print(f"    {period:15s}: mean={subset.mean():.3f}, "
              f"median={subset.median():.3f}, std={subset.std():.3f}, n={len(subset)}")

# ── 2.2 Kruskal-Wallis + post-hoc Mann-Whitney ──
print("\n--- 2.2 Kruskal-Wallis Tests (3-group comparison) ---")
for comp in COMPOSITES:
    groups = [df[df['covid_period'] == p][comp].dropna() for p in ORDER]
    H, p_kw = stats.kruskal(*groups)
    sig = '***' if p_kw < 0.001 else '**' if p_kw < 0.01 else '*' if p_kw < 0.05 else 'n.s.'
    print(f"\n  {comp}: H={H:.2f}, p={p_kw:.4f} {sig}")

    # Post-hoc pairwise
    pairs = [('pre_covid', 'during_covid'),
             ('during_covid', 'post_covid'),
             ('pre_covid', 'post_covid')]
    for p1, p2 in pairs:
        g1 = df[df['covid_period'] == p1][comp].dropna()
        g2 = df[df['covid_period'] == p2][comp].dropna()
        U, p_mw = stats.mannwhitneyu(g1, g2, alternative='two-sided')
        r = 1 - (2 * U) / (len(g1) * len(g2))  # rank-biserial
        sig_mw = '***' if p_mw < 0.001 else '**' if p_mw < 0.01 else '*' if p_mw < 0.05 else 'n.s.'
        print(f"    {p1:15s} vs {p2:15s}: U={U:.0f}, p={p_mw:.4f} {sig_mw}, r={r:+.3f}")

# ── 2.3 Individual service satisfaction trends ──
print("\n--- 2.3 Individual Service Satisfaction Trends (DS_) ---")
for col in CORE_DS:
    name = DS_NAMES.get(col, col)
    vals = []
    for period in ORDER:
        subset = df[df['covid_period'] == period][col].dropna()
        vals.append(subset.mean())
    delta = vals[2] - vals[0]
    print(f"  {name:25s} | Pre: {vals[0]:.2f} | During: {vals[1]:.2f} | "
          f"Post: {vals[2]:.2f} | Δ={delta:+.2f}")

# ── 2.4 Individual service usage trends ──
print("\n--- 2.4 Individual Service Usage Trends (USE_) ---")
for col in CORE_USE:
    name = USE_NAMES.get(col, col)
    vals = []
    for period in ORDER:
        subset = df[df['covid_period'] == period][col].dropna()
        vals.append(subset.mean())
    delta = vals[2] - vals[0]
    print(f"  {name:25s} | Pre: {vals[0]:.2f} | During: {vals[1]:.2f} | "
          f"Post: {vals[2]:.2f} | Δ={delta:+.2f}")

# ── 2.5 Dissatisfaction rates (% scoring 1 or 2) ──
print("\n--- 2.5 Dissatisfaction Rates (% scoring 1 or 2) ---")
for col in CORE_DS:
    name = DS_NAMES.get(col, col)
    parts = []
    for period in ORDER:
        vals = df[df['covid_period'] == period][col].dropna()
        pct = (vals <= 2).mean() * 100
        parts.append(f"{pct:5.1f}%")
    print(f"  {name:20s} | Pre: {parts[0]} | During: {parts[1]} | Post: {parts[2]}")

# ── 2.6 Importance-satisfaction gap by service (pre vs post) ──
print("\n--- 2.6 Service-Level Imp-Sat Gap (Pre vs Post COVID) ---")
pre = df[df['covid_period'] == 'pre_covid']
post = df[df['covid_period'] == 'post_covid']
for imp_col, ds_col, name in IMP_SAT_PAIRS:
    pre_gap = pre[imp_col].mean() - pre[ds_col].mean()
    post_gap = post[imp_col].mean() - post[ds_col].mean()
    print(f"  {name:15s} | Pre gap: {pre_gap:+.2f} | Post gap: {post_gap:+.2f}")

# ── 2.7 % of students with unmet needs ──
print("\n--- 2.7 % Students with Positive Gap (Importance > Satisfaction) ---")
for period in ORDER:
    gap = df[df['covid_period'] == period]['IT_importance_satisfaction_gap']
    pos_pct = (gap > 0).sum() / gap.dropna().shape[0] * 100
    neg_pct = (gap < 0).sum() / gap.dropna().shape[0] * 100
    print(f"  {period:15s}: Underserved={pos_pct:.1f}% | Over-satisfied={neg_pct:.1f}%")

# ── 2.8 Awareness distribution ──
print("\n--- 2.8 IT Awareness (INF_ATS) Distribution ---")
for period in ORDER:
    vals = df[df['covid_period'] == period]['INF_ATS'].dropna()
    print(f"\n  {period} (mean={vals.mean():.2f}):")
    for v, label in [(1, 'Not Informed'), (2, 'Somewhat'), (3, 'Informed'), (4, 'Very Informed')]:
        pct = ((vals == v).sum() / len(vals)) * 100
        print(f"    {label:15s}: {pct:5.1f}%")

# ── 2.9 Awareness × Satisfaction correlation ──
print("\n--- 2.9 Awareness × Satisfaction (Post-COVID) ---")
post_df = df[df['covid_period'] == 'post_covid']
for level in [1, 2, 3, 4]:
    subset = post_df[post_df['INF_ATS'] == level]['composite_IT_satisfaction']
    label = {1: 'Not Informed', 2: 'Somewhat', 3: 'Informed', 4: 'Very Informed'}[level]
    if len(subset) >= 5:
        print(f"  {label:15s}: mean sat = {subset.mean():.3f} (n={len(subset)})")
r_aw, p_aw = stats.spearmanr(
    post_df['INF_ATS'].dropna(),
    post_df.loc[post_df['INF_ATS'].notna(), 'composite_IT_satisfaction'].dropna()
)
print(f"  Spearman r = {r_aw:+.3f}, p = {p_aw:.4f}")

# ── 2.10 Usage × Satisfaction correlation by period ──
print("\n--- 2.10 Usage × Satisfaction Correlation by Period ---")
for period in ORDER:
    subset = df[df['covid_period'] == period].dropna(
        subset=['composite_IT_usage', 'composite_IT_satisfaction'])
    r, p = stats.spearmanr(subset['composite_IT_usage'],
                           subset['composite_IT_satisfaction'])
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'
    print(f"  {PERIOD_LABELS_INLINE[period]}: r={r:+.3f}, p={p:.4f} {sig}, n={len(subset)}")

# ── 2.11 Gender breakdown ──
print("\n--- 2.11 Gender Breakdown (Satisfaction) ---")
for period in ORDER:
    subset = df[df['covid_period'] == period]
    genders = subset.groupby('gender')['composite_IT_satisfaction'].agg(['mean', 'count'])
    genders = genders[genders['count'] >= 10]
    print(f"  {period}:")
    for g, row in genders.iterrows():
        print(f"    {g:15s}: mean={row['mean']:.3f} (n={int(row['count'])})")

# ── 2.12 Outlier sensitivity ──
print("\n--- 2.12 Outlier Sensitivity Check ---")
for comp in ['composite_IT_usage', 'composite_IT_satisfaction']:
    print(f"\n  {comp}:")
    for period in ORDER:
        subset = df[df['covid_period'] == period]
        all_mean = subset[comp].mean()
        no_out = subset[~subset['is_outlier_composite']][comp].mean()
        diff = no_out - all_mean
        print(f"    {period:15s}: All={all_mean:.3f}, No outliers={no_out:.3f}, Diff={diff:+.3f}")

# ── 2.13 Helpdesk staff dimensions ──
print("\n--- 2.13 Helpdesk Staff Individual Dimensions ---")
for col, name in DAHD_NAMES.items():
    vals = [df[df['covid_period'] == p][col].dropna().mean() for p in ORDER]
    print(f"  {name:25s} | Pre: {vals[0]:.2f} | During: {vals[1]:.2f} | Post: {vals[2]:.2f}")


# ══════════════════════════════════════════════════════════════
# 3. VISUALIZATION GENERATION
# ══════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("GENERATING VISUALIZATIONS")
print("=" * 70 + "\n")

# ── FIGURE 0: Dataset Overview ──
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

counts = df.groupby('covid_period').size().reindex(ORDER)
colors = [PERIOD_COLORS[p] for p in ORDER]
bars = axes[0].bar(range(3), counts.values, color=colors, width=0.6,
                   edgecolor='white', linewidth=1.5)
for bar, val in zip(bars, counts.values):
    axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                 str(val), ha='center', fontsize=12, fontweight='bold')
axes[0].set_xticks(range(3))
axes[0].set_xticklabels([PERIOD_LABELS[p] for p in ORDER], fontsize=9)
axes[0].set_ylabel('Number of Respondents')
axes[0].set_title('Survey Responses by Period', fontsize=12, fontweight='bold')
axes[0].set_ylim(0, 380)

genders = df.groupby(['covid_period', 'gender']).size().unstack(fill_value=0).reindex(ORDER)
g_cols = ['Female', 'Male', 'Not_reported']
g_colors = ['#E67E22', BLUE, GREY]
x = np.arange(3)
w = 0.25
for i, (gc, gcol) in enumerate(zip(g_cols, g_colors)):
    if gc in genders.columns:
        axes[1].bar(x + i * w, genders[gc].values, w, label=gc, color=gcol, edgecolor='white')
axes[1].set_xticks(x + w)
axes[1].set_xticklabels([PERIOD_LABELS[p] for p in ORDER], fontsize=9)
axes[1].set_ylabel('Count')
axes[1].set_title('Gender Distribution', fontsize=12, fontweight='bold')
axes[1].legend(fontsize=8)

comp_cols = ['composite_IT_usage', 'composite_IT_satisfaction', 'composite_IT_importance',
             'composite_IT_awareness', 'digital_engagement_index', 'composite_helpdesk_staff']
comp_labels_short = ['Usage', 'Satisfaction', 'Importance', 'Awareness', 'Digital Eng.', 'Helpdesk']
missing_pct = [(df[c].isna().sum() / len(df)) * 100 for c in comp_cols]
axes[2].barh(range(len(comp_cols)), missing_pct, color=BLUE, height=0.5)
for i, pct in enumerate(missing_pct):
    axes[2].text(pct + 0.3, i, f'{pct:.1f}%', va='center', fontsize=9)
axes[2].set_yticks(range(len(comp_cols)))
axes[2].set_yticklabels(comp_labels_short, fontsize=9)
axes[2].set_xlabel('% Missing')
axes[2].set_title('Composite Score Completeness', fontsize=12, fontweight='bold')
axes[2].set_xlim(0, 12)
axes[2].invert_yaxis()

fig.suptitle('Dataset Overview: 815 Student Respondents Across Three COVID Periods',
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
savefig(fig, 'eda_00_dataset_overview.png')


# ── FIGURE 1: Central Paradox ──
fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

data_sets = [
    ('composite_IT_usage', 'IT Service Usage', '1–5 scale', [1, 5]),
    ('composite_IT_satisfaction', 'IT Satisfaction', '1–4 scale', [1, 4]),
    ('composite_IT_awareness', 'IT Awareness', '1–4 scale', [1, 4]),
]
for ax, (col, title, ylabel, ylim) in zip(axes, data_sets):
    means = [df[df['covid_period'] == p][col].mean() for p in ORDER]
    colors_b = [PERIOD_COLORS[p] for p in ORDER]
    bars = ax.bar(range(3), means, color=colors_b, width=0.65,
                  edgecolor='white', linewidth=1.5)
    for i, (bar, val) in enumerate(zip(bars, means)):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (ylim[1] - ylim[0]) * 0.02,
                f'{val:.2f}', ha='center', va='bottom',
                fontsize=12, fontweight='bold', color=colors_b[i])
    ax.set_title(title, fontsize=14, fontweight='bold', pad=12)
    ax.set_ylabel(ylabel, fontsize=9, color=GREY)
    ax.set_xticks(range(3))
    ax.set_xticklabels([PERIOD_LABELS[p] for p in ORDER], fontsize=9)
    ax.set_ylim(ylim[0], ylim[1])
    delta = means[2] - means[0]
    ax.annotate(f'Δ = {delta:+.2f}', xy=(2.4, means[2]), fontsize=10,
                fontweight='bold', color=GREEN if delta > 0 else RED, ha='left')

fig.suptitle('The Central Paradox: Usage Declined While Satisfaction Improved',
             fontsize=16, fontweight='bold', y=1.02)
fig.text(0.5, -0.02,
         'MISO Student Survey — Framingham State University — IT Services Only (n=815)',
         ha='center', fontsize=9, color=GREY)
plt.tight_layout()
savefig(fig, 'eda_01_central_paradox.png')


# ── FIGURE 2: Service Satisfaction Slopes ──
fig, ax = plt.subplots(figsize=(12, 7))

x_pos = [0, 1, 2]
for col in CORE_DS:
    means = [df[df['covid_period'] == p][col].mean() for p in ORDER]
    delta = means[2] - means[0]
    name = DS_NAMES.get(col, col)
    if delta > 0.3:
        color, alpha, lw = GREEN, 0.9, 2.5
    elif delta < -0.1:
        color, alpha, lw = RED, 0.8, 2.5
    else:
        color, alpha, lw = GREY, 0.4, 1.5
    ax.plot(x_pos, means, '-o', color=color, alpha=alpha, linewidth=lw, markersize=6)
    ax.text(2.08, means[2], f'{name} ({delta:+.2f})', va='center', fontsize=8.5,
            color=color, fontweight='bold' if abs(delta) > 0.1 else 'normal')

ax.set_xticks(x_pos)
ax.set_xticklabels([PERIOD_LABELS_INLINE[p] for p in ORDER], fontsize=11)
ax.set_ylabel('Mean Satisfaction (1–4 scale)', fontsize=11)
ax.set_ylim(2.4, 4.1)
ax.set_xlim(-0.3, 3.8)
ax.set_title('Individual IT Service Satisfaction Trends (2018–2024)',
             fontsize=14, fontweight='bold', pad=15)
ax.plot([], [], '-', color=GREEN, linewidth=2.5, label='Major improvement (Δ > +0.30)')
ax.plot([], [], '-', color=RED, linewidth=2.5, label='Declined (Δ < -0.10)')
ax.plot([], [], '-', color=GREY, linewidth=1.5, label='Stable')
ax.legend(loc='lower left', fontsize=9, frameon=True, facecolor='white', edgecolor=GREY)
plt.tight_layout()
savefig(fig, 'eda_02_service_satisfaction_slopes.png')


# ── FIGURE 3: Banner Collapse ──
fig, axes = plt.subplots(1, 3, figsize=(14, 5))

for ax, period in zip(axes, ORDER):
    vals = df[df['covid_period'] == period]['USE_ERPSS'].dropna()
    dist = vals.value_counts(normalize=True).sort_index() * 100
    bars_data = [dist.get(v, 0) for v in [1, 2, 3, 4, 5]]
    bar_colors = ['#E74C3C' if v <= 2 else AMBER if v == 3 else GREEN for v in [1, 2, 3, 4, 5]]
    bars = ax.barh(range(5), bars_data, color=bar_colors, height=0.6, edgecolor='white')
    for i, (bar, pct) in enumerate(zip(bars, bars_data)):
        if pct > 3:
            ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                    f'{pct:.1f}%', va='center', fontsize=10, fontweight='bold')
    ax.set_yticks(range(5))
    ax.set_yticklabels([USE_FREQ_LABELS[v] for v in [1, 2, 3, 4, 5]], fontsize=9)
    ax.set_xlim(0, 80)
    ax.set_xlabel('% of Students', fontsize=9)
    ax.set_title(PERIOD_LABELS[period].replace('\n', ' '), fontsize=12,
                 fontweight='bold', color=PERIOD_COLORS[period])
    ax.invert_yaxis()

fig.suptitle('Banner/MyFramingham Usage Collapsed Post-COVID',
             fontsize=14, fontweight='bold', y=1.02)
fig.text(0.5, -0.02,
         'USE_ERPSS: 64.6% used 3+x/week pre-COVID → 46.1% never use it post-COVID',
         ha='center', fontsize=10, color=RED, fontweight='bold')
plt.tight_layout()
savefig(fig, 'eda_03_banner_collapse.png')


# ── FIGURE 4: WiFi Turnaround ──
fig, ax = plt.subplots(figsize=(10, 5.5))

wifi_data = {
    'WiFi Performance': [df[df['covid_period'] == p]['DS_PWAC'].mean() for p in ORDER],
    'WiFi Availability': [df[df['covid_period'] == p]['DS_AWAC'].mean() for p in ORDER],
}
x = np.arange(3)
width = 0.3
colors_wifi = ['#E74C3C', '#3498DB']
for i, (label, vals) in enumerate(wifi_data.items()):
    bars = ax.bar(x + i * width, vals, width, label=label, color=colors_wifi[i],
                  edgecolor='white', linewidth=1.5)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.03,
                f'{val:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

ax.set_xticks(x + width / 2)
ax.set_xticklabels([PERIOD_LABELS_INLINE[p] for p in ORDER], fontsize=11)
ax.set_ylabel('Mean Satisfaction (1–4 scale)', fontsize=11)
ax.set_ylim(1, 4.2)
ax.set_title('WiFi Satisfaction: The Biggest Improvement Story',
             fontsize=14, fontweight='bold', pad=15)
ax.legend(fontsize=10)
ax.annotate('WiFi Performance\n+0.71 improvement', xy=(2, 3.41), xytext=(2.3, 2.4),
            fontsize=9, fontweight='bold', color=GREEN,
            arrowprops=dict(arrowstyle='->', color=GREEN, lw=1.5))
plt.tight_layout()
savefig(fig, 'eda_04_wifi_turnaround.png')


# ── FIGURE 5: Gap Evolution ──
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

pre_data = df[df['covid_period'] == 'pre_covid']
post_data = df[df['covid_period'] == 'post_covid']
names = [p[2] for p in IMP_SAT_PAIRS]
pre_gaps = [pre_data[p[0]].mean() - pre_data[p[1]].mean() for p in IMP_SAT_PAIRS]
post_gaps = [post_data[p[0]].mean() - post_data[p[1]].mean() for p in IMP_SAT_PAIRS]

y = np.arange(len(names))
axes[0].barh(y - 0.2, pre_gaps, 0.35, label='Pre-COVID (2018)', color=BLUE, alpha=0.8)
axes[0].barh(y + 0.2, post_gaps, 0.35, label='Post-COVID (2024)', color=GREEN, alpha=0.8)
axes[0].axvline(x=0, color='black', linewidth=0.8)
axes[0].set_yticks(y)
axes[0].set_yticklabels(names, fontsize=10)
axes[0].set_xlabel('Importance − Satisfaction Gap', fontsize=10)
axes[0].set_title('Service-Level Gaps:\nPre vs Post-COVID', fontsize=12, fontweight='bold')
axes[0].legend(fontsize=9, loc='lower right')
axes[0].text(-0.5, -1.2, '← Over-satisfied', fontsize=8, color=GREY, ha='center')
axes[0].text(0.5, -1.2, 'Underserved →', fontsize=8, color=RED, ha='center')

gap_pcts = []
for period in ORDER:
    gap = df[df['covid_period'] == period]['IT_importance_satisfaction_gap']
    gap_pcts.append((gap > 0).sum() / gap.dropna().shape[0] * 100)
bars = axes[1].bar(range(3), gap_pcts, color=[PERIOD_COLORS[p] for p in ORDER],
                   width=0.6, edgecolor='white', linewidth=1.5)
for bar, val in zip(bars, gap_pcts):
    axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f'{val:.1f}%', ha='center', fontsize=12, fontweight='bold')
axes[1].set_xticks(range(3))
axes[1].set_xticklabels([PERIOD_LABELS[p] for p in ORDER], fontsize=9)
axes[1].set_ylabel('% of Students', fontsize=10)
axes[1].set_title('Students With Unmet Needs\n(Importance > Satisfaction)',
                   fontsize=12, fontweight='bold')
axes[1].set_ylim(0, 50)

fig.suptitle('The Satisfaction Surplus: From Service Gaps to Over-Delivery',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
savefig(fig, 'eda_05_gap_evolution.png')


# ── FIGURE 6: Awareness Distribution ──
fig, ax = plt.subplots(figsize=(10, 5.5))

awareness_labels = ['Not Informed', 'Somewhat\nInformed', 'Informed', 'Very\nInformed']
x = np.arange(4)
width = 0.25
for i, period in enumerate(ORDER):
    vals = df[df['covid_period'] == period]['INF_ATS'].dropna()
    dist = [((vals == v).sum() / len(vals)) * 100 for v in [1, 2, 3, 4]]
    ax.bar(x + i * width, dist, width,
           label=PERIOD_LABELS_INLINE[period],
           color=PERIOD_COLORS[period], alpha=0.85, edgecolor='white')

ax.set_xticks(x + width)
ax.set_xticklabels(awareness_labels, fontsize=10)
ax.set_ylabel('% of Students', fontsize=11)
ax.set_title('IT Service Awareness: The Shift from "Somewhat" to "Informed"',
             fontsize=14, fontweight='bold', pad=15)
ax.legend(fontsize=9)
ax.annotate('"Somewhat Informed"\ndropped from 57% to 37%', xy=(1, 56), xytext=(2.2, 50),
            fontsize=9, color=RED, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=RED, lw=1.5))
ax.annotate('"Informed" rose\nfrom 27% to 53%', xy=(2.3, 53), xytext=(3, 42),
            fontsize=9, color=GREEN, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=GREEN, lw=1.5))
plt.tight_layout()
savefig(fig, 'eda_06_awareness_shift.png')


# ── FIGURE 7: Composite Heatmap ──
fig, ax = plt.subplots(figsize=(10, 5))

metrics_hm = {
    'IT Usage (1-5)': 'composite_IT_usage',
    'IT Satisfaction (1-4)': 'composite_IT_satisfaction',
    'IT Importance (1-4)': 'composite_IT_importance',
    'IT Awareness (1-4)': 'composite_IT_awareness',
    'Digital Engagement (1-5)': 'digital_engagement_index',
    'Helpdesk Staff (1-4)': 'composite_helpdesk_staff',
}
heatmap_data = []
for name, col in metrics_hm.items():
    heatmap_data.append([df[df['covid_period'] == p][col].mean() for p in ORDER])

heatmap_df = pd.DataFrame(heatmap_data, index=list(metrics_hm.keys()),
                           columns=['Pre-COVID', 'During-COVID', 'Post-COVID'])

for i, (idx, row) in enumerate(heatmap_df.iterrows()):
    norm = Normalize(vmin=row.min() - 0.1, vmax=row.max() + 0.1)
    for j, val in enumerate(row):
        color_val = cm.RdYlGn(norm(val))
        ax.add_patch(plt.Rectangle((j, len(metrics_hm) - 1 - i), 1, 1,
                                   facecolor=color_val, edgecolor='white', linewidth=2))
        ax.text(j + 0.5, len(metrics_hm) - 1 - i + 0.5, f'{val:.2f}',
                ha='center', va='center', fontsize=13, fontweight='bold',
                color='white' if norm(val) < 0.3 or norm(val) > 0.8 else 'black')

ax.set_xlim(0, 3)
ax.set_ylim(0, len(metrics_hm))
ax.set_xticks([0.5, 1.5, 2.5])
ax.set_xticklabels([PERIOD_LABELS_INLINE[p] for p in ORDER], fontsize=11)
ax.set_yticks([i + 0.5 for i in range(len(metrics_hm))])
ax.set_yticklabels(list(reversed(list(metrics_hm.keys()))), fontsize=10)
ax.set_title('Composite Score Heatmap: The Full Picture',
             fontsize=14, fontweight='bold', pad=15)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.tick_params(left=False, bottom=False)
plt.tight_layout()
savefig(fig, 'eda_07_composite_heatmap.png')


# ── FIGURE 8: Canvas Adoption ──
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

for ax, period in zip(axes, ORDER):
    vals = df[df['covid_period'] == period]['USE_CMS'].dropna()
    dist = [((vals == v).sum() / len(vals)) * 100 for v in [1, 2, 3, 4, 5]]
    colors_bar = ['#E74C3C', '#E67E22', AMBER, '#3498DB', GREEN]
    bars = ax.bar(range(5), dist, color=colors_bar, width=0.7, edgecolor='white')
    for bar, pct in zip(bars, dist):
        if pct > 2:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f'{pct:.0f}%', ha='center', fontsize=10, fontweight='bold')
    ax.set_xticks(range(5))
    ax.set_xticklabels(['Never', '1-2x\n/sem', '1-3x\n/mo', '1-3x\n/wk', '3+x\n/wk'], fontsize=8)
    ax.set_ylim(0, 105)
    ax.set_title(PERIOD_LABELS[period].replace('\n', ' '), fontsize=12,
                 fontweight='bold', color=PERIOD_COLORS[period])
    ax.set_ylabel('% of Students' if period == 'pre_covid' else '', fontsize=9)

fig.suptitle('Canvas/LMS Usage: From Dominant to Near-Universal (98.4% use 3+x/week in 2024)',
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
savefig(fig, 'eda_08_canvas_adoption.png')


# ── FIGURE 9: Dissatisfaction Rate Trends ──
fig, ax = plt.subplots(figsize=(12, 6))

dissat_services = ['DS_PWAC', 'DS_AWAC', 'DS_CMSS', 'DS_CMS', 'DS_OCS',
                   'DS_ERPSS', 'DS_FPC', 'DS_SERPP']
dissat_names = {
    'DS_PWAC': 'WiFi Performance', 'DS_AWAC': 'WiFi Availability',
    'DS_CMSS': 'LMS Support', 'DS_CMS': 'Canvas/LMS',
    'DS_OCS': 'Overall Computing', 'DS_ERPSS': 'Banner',
    'DS_FPC': 'Help Desk', 'DS_SERPP': 'Banner Support'
}
x_pos = [0, 1, 2]
for col in dissat_services:
    rates = []
    for period in ORDER:
        vals = df[df['covid_period'] == period][col].dropna()
        rates.append((vals <= 2).mean() * 100)
    name = dissat_names.get(col, col)
    delta = rates[2] - rates[0]
    if abs(delta) > 5:
        color, alpha, lw = (GREEN if delta < 0 else RED), 0.9, 2.5
    else:
        color, alpha, lw = GREY, 0.4, 1.5
    ax.plot(x_pos, rates, '-o', color=color, alpha=alpha, linewidth=lw, markersize=6)
    ax.text(2.08, rates[2], f'{name} ({delta:+.1f}pp)', va='center', fontsize=8.5,
            color=color, fontweight='bold' if abs(delta) > 5 else 'normal')

ax.set_xticks(x_pos)
ax.set_xticklabels([PERIOD_LABELS_INLINE[p] for p in ORDER], fontsize=11)
ax.set_ylabel('% Dissatisfied (Score 1 or 2)', fontsize=11)
ax.set_title('Dissatisfaction Rates by Service: Who Got Better, Who Got Worse',
             fontsize=14, fontweight='bold', pad=15)
ax.set_xlim(-0.3, 4.2)
ax.set_ylim(-2, 42)
plt.tight_layout()
savefig(fig, 'eda_09_dissatisfaction_trends.png')


# ── FIGURE 10: Helpdesk Staff Stability ──
fig, ax = plt.subplots(figsize=(10, 5))

dahd_colors = ['#2980B9', '#27AE60', '#E67E22', '#8E44AD']
x = np.arange(3)
w = 0.18
for i, (col, name) in enumerate(DAHD_NAMES.items()):
    vals = [df[df['covid_period'] == p][col].mean() for p in ORDER]
    bars = ax.bar(x + i * w, vals, w, label=name, color=dahd_colors[i], edgecolor='white')
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{val:.2f}', ha='center', fontsize=8, fontweight='bold')

ax.set_xticks(x + 1.5 * w)
ax.set_xticklabels([PERIOD_LABELS_INLINE[p] for p in ORDER], fontsize=10)
ax.set_ylabel('Mean Rating (1–4)', fontsize=11)
ax.set_ylim(3.4, 4.05)
ax.set_title('Helpdesk Staff Ratings: The Unshakable Anchor',
             fontsize=14, fontweight='bold', pad=15)
ax.legend(fontsize=9)
ax.axhline(y=3.75, color=GREY, linestyle='--', alpha=0.3)
plt.tight_layout()
savefig(fig, 'eda_10_helpdesk_stability.png')


# ── FIGURE 11: Awareness × Satisfaction ──
fig, ax = plt.subplots(figsize=(9, 5))

post_df = df[df['covid_period'] == 'post_covid'].copy()
awareness_map = {1: 'Not Informed', 2: 'Somewhat\nInformed', 3: 'Informed', 4: 'Very\nInformed'}
aw_levels = [1, 2, 3, 4]
sat_means = []
sat_ns = []
for lev in aw_levels:
    subset = post_df[post_df['INF_ATS'] == lev]['composite_IT_satisfaction']
    sat_means.append(subset.mean())
    sat_ns.append(len(subset))

bars = ax.bar(range(4), sat_means, color=[RED, AMBER, BLUE, GREEN],
              width=0.6, edgecolor='white', linewidth=1.5)
for bar, val, n in zip(bars, sat_means, sat_ns):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
            f'{val:.2f}\n(n={n})', ha='center', fontsize=10, fontweight='bold')

ax.set_xticks(range(4))
ax.set_xticklabels([awareness_map[v] for v in aw_levels], fontsize=10)
ax.set_ylabel('Mean IT Satisfaction (1–4)', fontsize=11)
ax.set_ylim(3.0, 4.0)
ax.set_title('Post-COVID: More Aware Students Are More Satisfied',
             fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('Self-Reported IT Awareness Level', fontsize=11)
plt.tight_layout()
savefig(fig, 'eda_11_awareness_satisfaction.png')


# ── FIGURE 12: Usage by Individual Service ──
fig, ax = plt.subplots(figsize=(12, 5.5))

x = np.arange(len(CORE_USE))
w = 0.25
for i, period in enumerate(ORDER):
    vals = [df[df['covid_period'] == period][col].mean() for col in CORE_USE]
    ax.bar(x + i * w, vals, w, label=PERIOD_LABELS_INLINE[period],
           color=PERIOD_COLORS[period], edgecolor='white')

ax.set_xticks(x + w)
ax.set_xticklabels([USE_NAMES[c] for c in CORE_USE], fontsize=10)
ax.set_ylabel('Mean Usage Frequency (1–5)', fontsize=11)
ax.set_ylim(0, 5.5)
ax.set_title('IT Service Usage by Individual Service Across Periods',
             fontsize=14, fontweight='bold', pad=15)
ax.legend(fontsize=9)
plt.tight_layout()
savefig(fig, 'eda_12_usage_by_service.png')


# ── FIGURE 13: Usage vs Satisfaction Scatter (The Decoupling Effect) ──
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for ax, period in zip(axes, ORDER):
    subset = df[df['covid_period'] == period].dropna(
        subset=['composite_IT_usage', 'composite_IT_satisfaction'])
    x_vals = subset['composite_IT_usage']
    y_vals = subset['composite_IT_satisfaction']

    r, p_val = stats.spearmanr(x_vals, y_vals)

    # Scatter with slight jitter for readability
    ax.scatter(x_vals + np.random.normal(0, 0.03, len(x_vals)),
               y_vals + np.random.normal(0, 0.02, len(y_vals)),
               alpha=0.35, s=30, color=PERIOD_COLORS[period],
               edgecolors='white', linewidth=0.3)

    # Trend line
    z = np.polyfit(x_vals, y_vals, 1)
    poly = np.poly1d(z)
    x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
    ax.plot(x_line, poly(x_line), '--', color='black', linewidth=1.5, alpha=0.7)

    # Annotation box
    sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else ' (n.s.)'
    ax.text(0.05, 0.95, f'r = {r:+.3f}{sig}\nn = {len(subset)}',
            transform=ax.transAxes, fontsize=11, fontweight='bold',
            va='top', ha='left', color=PERIOD_COLORS[period],
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                      edgecolor=PERIOD_COLORS[period], alpha=0.9))

    ax.set_xlabel('IT Usage Composite (1–5)', fontsize=10)
    ax.set_ylabel('IT Satisfaction Composite (1–4)' if period == 'pre_covid' else '',
                  fontsize=10)
    ax.set_title(PERIOD_LABELS_INLINE[period], fontsize=12, fontweight='bold',
                 color=PERIOD_COLORS[period])
    ax.set_xlim(0.8, 5.2)
    ax.set_ylim(1.5, 4.2)

fig.suptitle('Usage vs. Satisfaction: The Decoupling Effect',
             fontsize=15, fontweight='bold', y=1.03)
fig.text(0.5, -0.03,
         'Each dot = one student. Dashed line = linear trend. Spearman correlation shown.',
         ha='center', fontsize=9, color=GREY)
plt.tight_layout()
savefig(fig, 'eda_13_usage_vs_satisfaction_scatter.png')


# ══════════════════════════════════════════════════════════════
# 4. SUMMARY
# ══════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("COMPLETE")
print("=" * 70)
print(f"\n14 figures saved to: {os.path.abspath(OUTPUT_DIR)}/")
print("""
Files generated:
  eda_00_dataset_overview.png          — Respondent counts, gender, completeness
  eda_01_central_paradox.png           — Usage vs satisfaction vs awareness bars
  eda_02_service_satisfaction_slopes.png — Slope chart: 12 services over 3 periods
  eda_03_banner_collapse.png           — Banner usage distribution shift
  eda_04_wifi_turnaround.png           — WiFi satisfaction improvement
  eda_05_gap_evolution.png             — Importance–satisfaction gap pre vs post
  eda_06_awareness_shift.png           — IT awareness distribution by period
  eda_07_composite_heatmap.png         — All composites at a glance
  eda_08_canvas_adoption.png           — Canvas usage distribution shift
  eda_09_dissatisfaction_trends.png    — % dissatisfied by service over time
  eda_10_helpdesk_stability.png        — Helpdesk staff ratings (4 dimensions)
  eda_11_awareness_satisfaction.png    — Awareness level × satisfaction crosstab
  eda_12_usage_by_service.png          — Individual service usage bars
  eda_13_usage_vs_satisfaction_scatter.png — Respondent-level scatter (decoupling)
""")
