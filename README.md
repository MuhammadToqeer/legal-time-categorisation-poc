# Legal Time Categorisation — POC

A proof-of-concept to automatically classify legal time “narratives” into business categories and surface billing/efficiency insights. Built to demonstrate an end-to-end ML lifecycle with clear business impact.

## Why this exists
- Manual categorisation is slow and inconsistent.
- ~75% entries are unlabelled → missed insights & delays.
- Goal: train a model to auto-classify narratives and quantify value (billable uplift, grade efficiency).

## Scope (POC)
- EDA → Feature engineering → Model benchmarking (A/B) → Explainability (SHAP/LIME) → Production plan.
- Simple Streamlit demo for single/batch predictions and interactive business insights.

## Data    

Columns: `Record ID, Department, Time Narrative, Worked Time, Charged to Client?, Grade, Category (partial)`.

## Tech
Python, scikit-learn, XGBoost/LightGBM, SHAP/LIME, Streamlit.

## Repo workflow
- `main` — stable, presentation-ready
- `dev` — integration branch
- Feature branches: `feature/eda`, `feature/modeling`, `feature/app`, `feature/explainability`
Commit style: `EDA: …`, `MODEL: …`, `APP: …`, `INF: …`

## How to run (will be added later)
- `requirements.txt`
- notebooks in `/notebooks`
- app in `/app` (Streamlit)
