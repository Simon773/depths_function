import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from data_utils import (
    NUMERIC_COLS,
    add_quality_flags,
    load_raw,
    station_equipment_profile,
)

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR = str(RESULTS_DIR)

df = load_raw()
df = add_quality_flags(df)
profile = station_equipment_profile(df)

print("Shape:", df.shape)
print("Stations:", df.number_sta.nunique(), "| Timestamps:", df.date.nunique())

# --- missing rate per variable ---
miss_rate = df[NUMERIC_COLS].isna().mean().sort_values(ascending=False)
print("\nMissing rate per variable:\n", (miss_rate * 100).round(1))

fig, ax = plt.subplots(figsize=(7, 4))
miss_rate.mul(100).plot(kind="barh", ax=ax, color="steelblue")
ax.set_xlabel("% missing")
ax.set_title("Missing data rate per variable")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/01_missing_rate.png", dpi=130)
plt.close()

# --- co-missingness pattern ---
miss_corr = df[NUMERIC_COLS].isna().corr()
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(miss_corr, annot=True, fmt=".2f", cmap="Reds", vmin=0, vmax=1, ax=ax)
ax.set_title("Co-missingness correlation (do variables go missing together?)")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/02_missing_cooccurrence.png", dpi=130)
plt.close()

# --- station equipment tiers ---
tier_counts = profile.groupby(
    ["has_wind", "has_humidity", "has_temp", "has_psl"]
).size()
tier_counts = tier_counts.sort_values(ascending=False)
print("\nStation equipment tiers (wind, humidity, temp, psl -> station count):")
print(tier_counts)
tier_counts.to_csv(f"{FIG_DIR}/station_equipment_tiers.csv")

# --- reporting completeness ---
fig, ax = plt.subplots(figsize=(6, 4))
profile["n_timestamps"].plot(kind="hist", bins=40, ax=ax, color="darkorange")
ax.set_xlabel("Number of reported timestamps out of 240")
ax.set_title("Station reporting completeness over the day")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/03_station_completeness.png", dpi=130)
plt.close()

profile.to_csv(f"{FIG_DIR}/station_profile.csv")
print("\nSaved figures and tables to", FIG_DIR)
