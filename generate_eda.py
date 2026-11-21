"""
Dengue EDA visuals for hackathon_preli_dataset.xlsx
Run: python generate_eda.py
Requires: pandas, matplotlib, openpyxl
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

INPUT = "hackathon_preli_dataset.xlsx"
OUT = "plots"
os.makedirs(OUT, exist_ok=True)

df = pd.read_excel(INPUT)
df["date"] = pd.to_datetime(df["date"])

# 1. Top 10 wards. Include city with ward_id to keep ward identifiers distinct.
ward_totals = (
    df.groupby(["city", "ward_id"], as_index=False)["dengue_cases"]
      .sum()
      .sort_values("dengue_cases", ascending=False)
      .head(10)
      .sort_values("dengue_cases")
)
ward_totals["ward_label"] = ward_totals["city"] + " — " + ward_totals["ward_id"]
plt.figure(figsize=(12, 7))
plt.barh(ward_totals["ward_label"], ward_totals["dengue_cases"])
plt.xlabel("Total dengue cases (2016–2025)")
plt.ylabel("City — Ward")
plt.title("Top 10 Hotspot Wards by Total Dengue Cases")
plt.grid(axis="x", alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "top_10_wards.png"), dpi=200, bbox_inches="tight")
plt.close()

# 2. Total cases by date (the source data is weekly).
weekly = (
    df.groupby("date", as_index=False)
      .agg(total_dengue_cases=("dengue_cases", "sum"),
           avg_rainfall_mm=("rainfall_mm", "mean"))
      .sort_values("date")
)
plt.figure(figsize=(14, 6))
plt.plot(weekly["date"], weekly["total_dengue_cases"], linewidth=1.5)
plt.xlabel("Date")
plt.ylabel("Total dengue cases")
plt.title("Weekly Dengue Cases Over Time (All Cities)")
plt.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "cases_over_time.png"), dpi=200, bbox_inches="tight")
plt.close()

# 3. Shift weekly mean rainfall by 1, 2 and 3 observations (weeks).
lagged = weekly.copy()
fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
for ax, lag in zip(axes, (1, 2, 3)):
    x = lagged["avg_rainfall_mm"].shift(lag)
    y = lagged["total_dengue_cases"]
    valid = x.notna() & y.notna()
    ax.scatter(x[valid], y[valid], alpha=0.45, s=18)
    corr = x[valid].corr(y[valid])
    if x[valid].nunique() > 1:
        m, b = np.polyfit(x[valid], y[valid], 1)
        xs = np.linspace(x[valid].min(), x[valid].max(), 100)
        ax.plot(xs, m * xs + b, linewidth=1.5)
    ax.set_title(f"{lag}-week lag (r={corr:.3f})")
    ax.set_xlabel(f"Mean rainfall {lag} week(s) earlier (mm)")
    ax.grid(alpha=0.25)
axes[0].set_ylabel("Current-week total dengue cases")
fig.suptitle("Rainfall Lag vs. Dengue Cases (Weekly, All Cities)", y=1.03)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "rainfall_lag_vs_cases.png"), dpi=200, bbox_inches="tight")
plt.close(fig)

print("Saved 3 charts in the plots/ folder.")
