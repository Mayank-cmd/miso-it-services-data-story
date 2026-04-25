"""
MISO Student Survey — Presentation Chart Generation
IT Services at Framingham State University
Pre-COVID (2018) · During-COVID (2021) · Post-COVID (2024)

Team 5 | MISM 6213 | Northeastern University | April 2026

Generates 7 presentation-quality charts following Dykes' 7 Principles:
  1. Right data        5. Focus attention
  2. Right viz         6. Make approachable
  3. Right config      7. Instill trust
  4. Remove noise

All bar charts start y-axis at 0 (instill trust).
All titles state the finding, not the data (focus attention).
No gridlines, no chartjunk (remove noise).
300 DPI for projection quality.

Usage:
    python presentation_charts.py

Requirements:
    pip install pandas numpy matplotlib scipy
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
import warnings
import os

warnings.filterwarnings('ignore')

# ── CONFIG ──
CSV_FILE = "../data/MISO_Students_IT_Analysis_Ready_v2.csv"
OUTPUT_DIR = "../figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

NAVY = '#1B4F72'
RED = '#C0392B'
GREEN = '#27874A'
AMBER = '#D4920B'
GREY = '#6B7280'
ORDER = ['pre_covid', 'during_covid', 'post_covid']
LABELS = ['Pre-COVID\n(2018)', 'During-COVID\n(2021)', 'Post-COVID\n(2024)']

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.grid': False,
})

# ── LOAD DATA ──
df = pd.read_csv(CSV_FILE, low_memory=False)
print(f"Loaded: {df.shape[0]} rows x {df.shape[1]} columns\n")


# ══════════════════════════════════════════════════════════════
# CHART 1: USAGE DECLINE
# Slide: "Usage Declined Across the Board"
# ══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 5.5))
usage = [df[df['covid_period']==p]['composite_IT_usage'].mean() for p in ORDER]
colors = [NAVY, RED, GREEN]
bars = ax.bar(range(3), usage, color=colors, width=0.55, edgecolor='white', linewidth=2)
for bar, val in zip(bars, usage):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.08, f'{val:.2f}',
            ha='center', fontsize=15, fontweight='bold', color=bar.get_facecolor())
ax.set_xticks(range(3)); ax.set_xticklabels(LABELS, fontsize=12)
ax.set_ylabel('Mean Usage Score (1-5)', fontsize=12)
ax.set_ylim(0, 5)
ax.set_title('IT Service Usage Has Dropped 20% and Never Recovered',
             fontsize=14, fontweight='bold', pad=15, color=NAVY)
ax.annotate('', xy=(2, usage[2]+0.15), xytext=(0, usage[0]+0.15),
            arrowprops=dict(arrowstyle='->', color=RED, lw=2.5))
ax.text(1, usage[0]+0.3, '-20%', ha='center', fontsize=18, fontweight='bold', color=RED)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'final_01_usage_decline.png'), dpi=300, bbox_inches='tight')
plt.close()
print("1/7 - Usage decline")


# ══════════════════════════════════════════════════════════════
# CHART 2: THE PARADOX
# Slide: "But Satisfaction Went Up"
# ══════════════════════════════════════════════════════════════
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

bars1 = ax1.bar(range(3), usage, color=[NAVY]*3, width=0.55, edgecolor='white', linewidth=2, alpha=0.85)
for bar, val in zip(bars1, usage):
    ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.08, f'{val:.2f}',
             ha='center', fontsize=14, fontweight='bold', color=NAVY)
ax1.set_xticks(range(3)); ax1.set_xticklabels(LABELS, fontsize=11)
ax1.set_ylabel('Mean Score', fontsize=12); ax1.set_ylim(0, 5)
ax1.set_title('Usage (1-5 scale)', fontsize=14, fontweight='bold', color=NAVY)
ax1.annotate('Down 20%', xy=(1.5, 4.3), fontsize=15, fontweight='bold', color=RED, ha='center')

sat = [df[df['covid_period']==p]['composite_IT_satisfaction'].mean() for p in ORDER]
bars2 = ax2.bar(range(3), sat, color=[GREEN]*3, width=0.55, edgecolor='white', linewidth=2, alpha=0.85)
for bar, val in zip(bars2, sat):
    ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.06, f'{val:.2f}',
             ha='center', fontsize=14, fontweight='bold', color=GREEN)
ax2.set_xticks(range(3)); ax2.set_xticklabels(LABELS, fontsize=11)
ax2.set_ylabel('Mean Score', fontsize=12); ax2.set_ylim(0, 4.5)
ax2.set_title('Satisfaction (1-4 scale)', fontsize=14, fontweight='bold', color=GREEN)
ax2.annotate('Up 6%', xy=(1.5, 4.0), fontsize=15, fontweight='bold', color=GREEN, ha='center')

fig.suptitle('The Paradox: Usage Fell While Satisfaction Rose',
             fontsize=16, fontweight='bold', y=1.02, color=NAVY)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'final_02_paradox.png'), dpi=300, bbox_inches='tight')
plt.close()
print("2/7 - Paradox")


# ══════════════════════════════════════════════════════════════
# CHART 3: THE AHA MOMENT — Canvas vs Banner Divergence
# Slide: "Students didn't disengage — they chose a winner"
# ══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 6))
canvas = [df[df['covid_period']==p]['USE_CMS'].mean() for p in ORDER]
banner = [df[df['covid_period']==p]['USE_ERPSS'].mean() for p in ORDER]

ax.plot(range(3), canvas, marker='o', markersize=12, linewidth=4,
        label='Canvas / LMS', color=NAVY, zorder=5)
ax.plot(range(3), banner, marker='s', markersize=12, linewidth=4,
        label='Banner Portal', color=RED, zorder=5)

for i, v in enumerate(canvas):
    ax.text(i, v+0.18, f'{v:.2f}', ha='center', fontsize=14, fontweight='bold', color=NAVY)
for i, v in enumerate(banner):
    ax.text(i, v-0.28, f'{v:.2f}', ha='center', fontsize=14, fontweight='bold', color=RED)

ax.annotate('Started nearly equal\nin 2018', xy=(0, 4.53), xytext=(0.5, 3.3),
            fontsize=11, color=GREY, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=GREY, lw=1.5))
ax.annotate('', xy=(2.12, canvas[2]), xytext=(2.12, banner[2]),
            arrowprops=dict(arrowstyle='<->', color=GREY, lw=2.5))
ax.text(2.25, (canvas[2]+banner[2])/2, f'Gap: {canvas[2]-banner[2]:.1f}',
        fontsize=13, fontweight='bold', color=GREY, va='center')

ax.set_xticks(range(3)); ax.set_xticklabels(LABELS, fontsize=13)
ax.set_ylabel('Mean Usage Score (1-5)', fontsize=13)
ax.set_ylim(1, 5.5)  # Line chart: non-zero baseline acceptable
ax.set_title('Students Consolidated onto Canvas - Banner Was Left Behind',
             fontsize=15, fontweight='bold', pad=15, color=NAVY)
ax.legend(fontsize=13, loc='center left')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'final_03_aha_divergence.png'), dpi=300, bbox_inches='tight')
plt.close()
print("3/7 - Aha divergence")


# ══════════════════════════════════════════════════════════════
# CHART 4: WIFI PROOF OF CONCEPT
# Slide: "Investment Works: The WiFi Turnaround"
# ══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(9, 5.5))
perf = [df[df['covid_period']==p]['DS_PWAC'].mean() for p in ORDER]
avail = [df[df['covid_period']==p]['DS_AWAC'].mean() for p in ORDER]
x = np.arange(3); w = 0.3

bars1 = ax.bar(x-w/2, perf, w, label='WiFi Performance', color=RED,
               edgecolor='white', linewidth=1.5, alpha=0.85)
bars2 = ax.bar(x+w/2, avail, w, label='WiFi Availability', color=NAVY,
               edgecolor='white', linewidth=1.5, alpha=0.85)

for bar, val in zip(bars1, perf):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.05, f'{val:.2f}',
            ha='center', fontsize=11, fontweight='bold', color=RED)
for bar, val in zip(bars2, avail):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.05, f'{val:.2f}',
            ha='center', fontsize=11, fontweight='bold', color=NAVY)

ax.set_xticks(x); ax.set_xticklabels(LABELS, fontsize=12)
ax.set_ylabel('Mean Satisfaction (1-4)', fontsize=12); ax.set_ylim(0, 4.5)
ax.set_title('WiFi Investment Worked: Satisfaction Jumped +0.71',
             fontsize=14, fontweight='bold', pad=15, color=NAVY)
ax.legend(fontsize=11)
ax.annotate('+0.71', xy=(2-w/2, perf[2]+0.15), fontsize=14, fontweight='bold', color=GREEN, ha='center')
ax.annotate('+0.68', xy=(2+w/2, avail[2]+0.15), fontsize=14, fontweight='bold', color=GREEN, ha='center')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'final_04_wifi_proof.png'), dpi=300, bbox_inches='tight')
plt.close()
print("4/7 - WiFi proof")


# ══════════════════════════════════════════════════════════════
# CHART 5: QUALITY GAP CLOSED
# Slide: "The Quality Gap Is Largely Closed"
# ══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 5.5))
gap_pcts = [
    (df[df['covid_period']==p]['IT_importance_satisfaction_gap']>0).sum() /
    df[df['covid_period']==p]['IT_importance_satisfaction_gap'].dropna().shape[0] * 100
    for p in ORDER
]

bars = ax.bar(range(3), gap_pcts, color=[NAVY, RED, GREEN], width=0.55, edgecolor='white', linewidth=2)
for bar, val in zip(bars, gap_pcts):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1.2, f'{val:.0f}%',
            ha='center', fontsize=16, fontweight='bold')
ax.set_xticks(range(3)); ax.set_xticklabels(LABELS, fontsize=12)
ax.set_ylabel('% of Students with Unmet Needs', fontsize=12); ax.set_ylim(0, 55)
ax.set_title('The Quality Gap Is Largely Closed',
             fontsize=14, fontweight='bold', pad=15, color=NAVY)
ax.annotate('', xy=(2, gap_pcts[2]+4), xytext=(0, gap_pcts[0]+4),
            arrowprops=dict(arrowstyle='->', color=GREEN, lw=2.5))
ax.text(1, gap_pcts[0]+7, 'Down from 35% to 16%',
        ha='center', fontsize=13, fontweight='bold', color=GREEN)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'final_05_gap_closed.png'), dpi=300, bbox_inches='tight')
plt.close()
print("5/7 - Gap closed")


# ══════════════════════════════════════════════════════════════
# CHART 6: AWARENESS — Lollipop Style
# Slide: "The Missing Piece: Awareness"
# ══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 5.5))
awareness = [df[df['covid_period']==p]['INF_ATS'].mean() for p in ORDER]

for i, (val, color) in enumerate(zip(awareness, [NAVY, RED, GREEN])):
    ax.plot(i, val, 'o', markersize=16, color=color, zorder=5)
    ax.vlines(i, 0, val, colors=color, linewidth=3, alpha=0.4)
    ax.text(i, val+0.1, f'{val:.2f}', ha='center', fontsize=15, fontweight='bold', color=color)

ax.axhline(y=3.0, color=GREY, linestyle='--', alpha=0.5, linewidth=1.5)
ax.text(2.4, 3.03, '"Informed"\nthreshold', fontsize=9, color=GREY, va='bottom')

ax.set_xticks(range(3)); ax.set_xticklabels(LABELS, fontsize=12)
ax.set_ylabel('Mean Awareness Score (1-4)', fontsize=12); ax.set_ylim(0, 4)
ax.set_title('Awareness Has Barely Moved in Six Years',
             fontsize=14, fontweight='bold', pad=15, color=NAVY)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'final_06_awareness.png'), dpi=300, bbox_inches='tight')
plt.close()
print("6/7 - Awareness")


# ══════════════════════════════════════════════════════════════
# CHART 7: HELPDESK STABILITY — Lollipop Style
# Slide: "It's Not a People Problem"
# ══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 5.5))
helpdesk = [df[df['covid_period']==p]['composite_helpdesk_staff'].mean() for p in ORDER]

for i, (val, color) in enumerate(zip(helpdesk, [NAVY, RED, GREEN])):
    ax.plot(i, val, 'o', markersize=16, color=color, zorder=5)
    ax.vlines(i, 0, val, colors=color, linewidth=3, alpha=0.4)
    ax.text(i, val+0.08, f'{val:.2f}', ha='center', fontsize=15, fontweight='bold', color=color)

ax.axhspan(3.75, 3.85, alpha=0.1, color=GREEN)
ax.text(2.4, 3.80, 'Consistently\nexcellent', fontsize=9, color=GREEN, va='center', fontweight='bold')

ax.set_xticks(range(3)); ax.set_xticklabels(LABELS, fontsize=12)
ax.set_ylabel('Mean Staff Rating (1-4)', fontsize=12); ax.set_ylim(0, 4.3)
ax.set_title("It's Not a People Problem - Helpdesk Ratings Are Rock Solid",
             fontsize=14, fontweight='bold', pad=15, color=NAVY)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'final_07_helpdesk.png'), dpi=300, bbox_inches='tight')
plt.close()
print("7/7 - Helpdesk")


print(f"\nAll 7 charts saved to {os.path.abspath(OUTPUT_DIR)}/")
print("Charts follow Dykes' 7 Principles:")
print("  - Bar charts start y-axis at 0 (instill trust)")
print("  - Line chart uses non-zero baseline (acceptable for trends)")
print("  - Titles state findings, not descriptions (focus attention)")
print("  - No gridlines or chartjunk (remove noise)")
print("  - DPI: 300 (projection quality)")
