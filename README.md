# SleepRAG Results Dashboard

Interactive dashboard for exploring the causal effect of adenotonsillectomy on pediatric sleep outcomes, estimated via propensity score matching on the NCH Sleep DataBank.

**Live app:** https://sleepragdashboard.streamlit.app

## What this shows

- **Forest plots** comparing treatment effects across four PSM specifications
- **Covariate balance loveplots** showing before vs after matching
- **Significance heatmap** across all outcomes and specifications
- **Dynamic KPIs** that update when you switch outcomes or specifications

## Data

The two CSV files in this repo are aggregate summary statistics (no patient-level data). They were produced by the full SleepRAG pipeline, which is hosted in a separate private repository.

## Run locally

```bash
pip install streamlit pandas numpy plotly
streamlit run sleeprag_dashboard.py
```
