from pathlib import Path
import pandas as pd

df = pd.read_excel(Path("hackathon_preli_dataset.xlsx"))
df.columns = [c.strip() for c in df.columns]
df["date"] = pd.to_datetime(df["date"], errors="coerce")

# 1. Check date spacing for a single ward
one_ward = df[df["ward_id"] == df["ward_id"].iloc[0]].sort_values("date")
print("Date differences for one ward:")
print(one_ward["date"].diff().value_counts().head(5))

# 2. Check overall date range and total rows
print("\nMin date:", df["date"].min(), "| Max date:", df["date"].max())
print("Total rows:", len(df), "| Unique wards:", df["ward_id"].nunique())