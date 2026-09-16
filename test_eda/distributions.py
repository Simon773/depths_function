import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from data_utils import NUMERIC_COLS, add_quality_flags, load_raw

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR = str(RESULTS_DIR)

df = load_raw()
df = add_quality_flags(df)

# --- histograms for continuous variables (excluding circular dd, sparse psl handled separately) ---
cont_vars = ["ff", "precip", "hu", "td", "t"]
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
for ax, col in zip(axes.flat, cont_vars):
    sns.histplot(df[col].dropna(), bins=60, ax=ax, color="teal")
    ax.set_title(col)
axes.flat[-1].axis("off")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/04_histograms.png", dpi=130)
plt.close()

# --- psl separately, it's nearly always missing but worth seeing its own range ---
fig, ax = plt.subplots(figsize=(6, 4))
sns.histplot(df["psl"].dropna(), bins=40, ax=ax, color="purple")
ax.set_title("psl (sea-level pressure, Pa) - only 12% of rows")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/05_psl_hist.png", dpi=130)
plt.close()

# --- precip: zero-inflated, look at nonzero part ---
nonzero_precip = df.loc[df["precip"] > 0, "precip"]
print(
    "Share of precip records that are exactly 0:", (df["precip"] == 0).mean().round(3)
)
print("Nonzero precip stats:\n", nonzero_precip.describe())

fig, ax = plt.subplots(figsize=(6, 4))
sns.histplot(nonzero_precip, bins=30, ax=ax, color="dodgerblue")
ax.set_title("precip distribution, nonzero values only (mm / 6 min)")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/06_precip_nonzero.png", dpi=130)
plt.close()

# --- wind direction: circular variable, needs a polar rose, not a linear histogram ---
dd = df["dd"].dropna()
ff = df.loc[df["dd"].notna(), "ff"]
theta = np.deg2rad(dd)
fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"projection": "polar"})
ax.set_theta_zero_location("N")
ax.set_theta_direction(-1)
hb = ax.hexbin(theta, ff, gridsize=25, cmap="viridis", mincnt=1)
ax.set_title("Wind rose: direction (dd) vs speed (ff)")
plt.colorbar(hb, ax=ax, label="count")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/07_wind_rose.png", dpi=130)
plt.close()

# --- outlier flags summary ---
print("\nQuality flags:")
print("hu_out_of_range:", df["hu_out_of_range"].sum())
print("dd_out_of_range:", df["dd_out_of_range"].sum())
print("ff_negative:", df["ff_negative"].sum())

# where do the hu>100 readings come from - few stations or spread out?
bad_hu_stations = df.loc[df["hu_out_of_range"], "number_sta"].value_counts()
print("\nStations contributing hu>100 readings (top 10):\n", bad_hu_stations.head(10))
print("Max hu value observed:", df["hu"].max())

print("\nSaved figures to", FIG_DIR)
