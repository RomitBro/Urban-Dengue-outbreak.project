# Urban Dengue Outbreak Prediction Pipeline

A ward-level dengue case forecasting system developed for the AI for Urban Dengue Outbreak Challenge. This project utilizes weekly spatio-temporal features and historical climate metrics to predict dengue cases across urban wards, aiding early intervention and resource allocation.

---

## 📌 Project Overview

Dengue fever outbreaks pose significant public health challenges in urban settings. This repository implements an end-to-end machine learning pipeline that models weekly ward-level case counts using temporal lag features and local environmental signals.

### Key Highlights
- **Data Cadence:** Weekly (`7-day` intervals) ward-level observations from **2016-01-04 to 2025-12-29** ($31,320$ total records).
- **Target Distribution:** Highly right-skewed count target ($58.03\%$ zero-case weeks, median $0.0$, mean $2.64$, max $41$ cases).
- **Domain Insight:** Incorporated a confirmed **3-week rainfall lag signal**, reflecting vector breeding dynamic cycles.
- **Leakage Prevention:** Features are engineered strictly grouped by `ward_id` with a temporal split to ensure zero future data leakage.

---

## 📊 Exploratory Data Analysis (EDA)

Key visual insights generated during EDA (located in `feature/eda-visuals`):

1. **Cases Over Time (`cases_over_time.png`):** Captures seasonal dengue spikes and long-term multi-year outbreak cycles.
2. **Rainfall Lag vs. Cases (`rainfall_lag_vs_cases.png`):** Demonstrates the strong delayed correlation between peak precipitation and subsequent case surges 3 weeks later.
3. **Top 10 Wards (`top_10_wards.png`):** Highlights high-risk geographic hotspots across the target municipality.

---

## ⚙️ Modeling & Validation Strategy

### Feature Engineering
- **Ward-Grouped Lags:** Historical dengue case lags and rolling statistics calculated per `ward_id`.
- **Precipitation Lags:** 3-week rainfall lags to model vector ecology.

### Validation Setup
To evaluate model performance realistically under real-world forecasting conditions, we use a strict chronological split:
- **Training Set:** `2016 – 2023` ($24,840$ rows)
- **Validation Set:** `2024 – 2025` ($6,300$ rows)

---

## 📈 Baseline Model Results

Our baseline pipeline utilizes a **`RandomForestRegressor`**:

| Metric | Score | Notes |
| :--- | :--- | :--- |
| **MAE** | **0.93** | Average absolute error per ward-week |
| **RMSE** | **1.77** | Penalizes larger outbreak spikes on skewed tail data |

---

## 📂 Repository Structure

```text
├── train.py              # Main training & validation pipeline
├── check_dates.py        # Date continuity & spacing verification script
├── generate_eda.py       # Visualizations & plot generator script
├── cases_over_time.png   # EDA plot: Case timeline
├── rainfall_lag_vs_cases.png # EDA plot: Rainfall lag correlation
├── top_10_wards.png      # EDA plot: Top affected wards breakdown
├── .gitignore            # Excludes dataset (*.csv) and local artifacts
└── README.md             # Project documentation

## 👥 Team Members & Roles
- **Nipun Barua:** Pipeline Architecture & Baseline Model
- **Mohana Bhowmik:** Data Analysis & Visualizations
- **[ Aditya Sarkar ]:** Validation Check & Demo Presentation
-
