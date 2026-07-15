# Hard Drive Reliability Analysis

Analysis of Backblaze's public hard drive failure dataset (Q1 2025) — connecting
to a Seagate Data Analytics internship background. Investigates which SMART
attributes are associated with drive failure.

## Status: Weeks 1-4 complete (demo pipeline); pending rerun on real Q1 2025 data

## Research question
Which SMART attributes are most associated with drive failure among drive models with
sufficient failure events (n >= 30), and can we distinguish failing drives in the days
before failure?

SMART stands for Self-Monitoring, Analysis, and Reporting Technology. A "drive dataset" refers to the continuous logs and internal metrics recorded by the drive's firmware to predict failures and monitor the storage device's overall health.

## Findings (draft — from demo pipeline, to be updated with real-data numbers)

- Attributes most associated with failure: reallocated sector count (`smart_5_raw`),
  reported uncorrectable errors (`smart_187_raw`), and current pending sector count
  (`smart_197_raw`) — all statistically significant (p < 0.05) with large effect sizes
- Power-on hours (`smart_9_raw`) served as a negative control and was **not**
  significantly associated with failure, as expected
- No attribute pairs showed correlation above 0.7, so no redundant attributes needed
  to be dropped from the analysis
- A simple logistic regression evaluation confirmed separable signal using just
  these attributes (see `notebooks/03_statistical_analysis.ipynb` for caveats on
  interpreting this — it's descriptive, not a final model)
- See `reports/figures/` for the full set of visuals (failure rate by model, boxplots,
  correlation heatmap, effect size summary)

**Important:** the numbers above come from a small synthetic sample used to validate
the pipeline end-to-end (`data/raw/sample/`). Rerun all four notebooks against the real
Q1 2025 data before treating any of these findings as final — see the checklist at the
end of each notebook for exactly what to revisit (`MIN_FAILURES`, `SPARSE_THRESHOLD`,
and the bracketed figures in `04_visualization_report.ipynb`, Section 5).

## Data
- Source: [Backblaze Hard Drive Stats](https://www.backblaze.com/cloud-storage/resources/hard-drive-test-data) — Q1 2025
- Not committed to this repo (see `.gitignore`); download separately and place
  daily CSVs in `data/raw/`
- A small synthetic sample mimicking the real schema lives in `data/raw/sample/`
  for demonstrating the pipeline without the full download

## Tooling
- **DuckDB** for querying raw CSVs directly (columnar, out-of-core — handles the
  full quarter without loading everything into memory at once)
- **pandas** for downstream stats/plotting once data is filtered down to a
  workable size
- **Jupyter** notebooks for the analysis workflow

## Repo structure
```
hard-drive-reliability-analysis/
├── data/
│   ├── raw/              # gitignored — never commit
│   └── processed/        # small derived files (parquet/csv)
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_cleaning_eda.ipynb
│   ├── 03_statistical_analysis.ipynb
│   └── 04_visualization_report.ipynb
├── src/
│   ├── load_data.py       # DuckDB loading/query functions
│   ├── cleaning.py
│   └── stats_utils.py
└── reports/figures/
```

## Setup
```bash
pip install -r requirements.txt
```

## Limitations
- Single quarter of data (Q1 2025) — no long-term drive age trends
- Failures are rare events (<1% of drive-days); statistical claims are scoped
  to models with sufficient failure counts (see Week 1 notebook)

## Next steps
- Rerun all four notebooks against the real Q1 2025 data (see per-notebook checklists
  for exactly what to revisit)
- Bridge to Month 4's ML project: survival analysis (time-to-failure) or a tuned
  classification model, using the same eligible-model scope and feature set
