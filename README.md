# From Investment to Impact: The IT Consolidation Story at Framingham State

**MISM 6213: Enterprise Data & Analytics Strategy | Northeastern University | Spring 2026**

## Overview

This project analyzes longitudinal student survey data from the [MISO (Measuring Information Service Outcomes)](https://www.misosurvey.org/) program at Framingham State University to uncover how student IT service engagement has evolved across three survey waves (2018, 2021, 2024). The analysis is framed as a data story for the university's Chief Information Officer (CIO).

## The Central Insight

> **Students didn't disengage from IT services — they consolidated onto a single platform.**
>
> Canvas usage rose to 98.4% daily adoption while 46.1% of students stopped using the Banner portal entirely. The 20% decline in composite IT usage is not disengagement — it is platform consolidation. The quality gap is closed. The next challenge is awareness, adoption, and consolidation.

## Key Findings

- **Usage-Satisfaction Paradox:** IT usage dropped 20% (3.25 → 2.60) while satisfaction rose 6% (3.46 → 3.66). Both statistically significant (p < 0.001).
- **Platform Divergence:** Canvas and Banner started at nearly identical usage in 2018 (4.56 vs 4.50). By 2024, a 2.73-point gap had opened.
- **WiFi Turnaround:** Targeted investment reduced WiFi dissatisfaction from 36% to 16% — proof the invest-and-measure model works.
- **Awareness Gap:** IT awareness has barely moved in six years (2.38 → 2.61 on a 4-point scale), and better-informed students are more satisfied (r = +0.21).

## Project Structure

```
miso-it-services-data-story/
├── README.md
├── data/
│   └── MISO_Students_IT_Analysis_Ready_v2.csv
├── scripts/
│   ├── miso_data_prep.py          # Data cleaning & preparation pipeline
│   ├── miso_patch.py              # Patch for encoding bugs & composite recalculation
│   ├── EDA_Complete.py            # Full exploratory data analysis (Spyder-optimized)
│   └── presentation_charts.py    # 7 presentation-quality chart generation
├── figures/
│   ├── final_01_usage_decline.png
│   ├── final_02_paradox.png
│   ├── final_03_aha_divergence.png
│   ├── final_04_wifi_proof.png
│   ├── final_05_gap_closed.png
│   ├── final_06_awareness.png
│   └── final_07_helpdesk.png
├── deliverables/
│   ├── MISO_Interim_Report.docx
│   ├── MISM6213_Data_Trailer_Team5.docx
│   └── MISO_Final_Presentation_Team5.pptx
└── .gitignore
```

## Methodology

- **Data Preparation:** Merged three Qualtrics CSV exports, handled encoding issues (Latin-1), converted sentinel values (-99 → NaN), applied median imputation for columns with <40% missingness, engineered six composite scores, flagged outliers using IQR method.
- **Statistical Tests:** Kruskal-Wallis H-tests for cross-period comparisons, Mann-Whitney U post-hoc tests, Spearman rank correlations, outlier sensitivity analysis.
- **Visualization Principles:** All charts follow Brent Dykes' 7 principles from *Effective Data Storytelling* — bar charts start at 0 (instill trust), titles state findings not descriptions (focus attention), no gridlines or chartjunk (remove noise), source citations and scale labels (make approachable).

## Tools & Technologies

- **Python** (pandas, NumPy, SciPy, matplotlib) via Spyder IDE
- **Microsoft PowerPoint** for final presentation
- **Canva** for visual design work

## Course References

- Dykes, B. (2020). *Effective Data Storytelling: How to Drive Change with Data, Narrative and Visuals.* Wiley.
- Wixom, Someh & Beath. *Data Is Everybody's Business.* MIT Press.
- MISO Survey Program: [misosurvey.org](https://www.misosurvey.org/)

## Team

Team 5 | MISM 6213 | Northeastern University | Spring 2026

