import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from data_utils import load_raw, station_equipment_profile

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR = str(RESULTS_DIR)

df = load_raw()
profile = station_equipment_profile(df)

# --- spatial: station network colored by altitude ---
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
sc = axes[0].scatter(
    profile["lon"], profile["lat"], c=profile["height_sta"], cmap="terrain", s=18
)
axes[0].set_title("Station network - colored by altitude (m)")
axes[0].set_xlabel("lon")
axes[0].set_ylabel("lat")
plt.colorbar(sc, ax=axes[0])

mean_t_by_station = df.groupby("number_sta")["t"].mean() - 273.15
axes[1].scatter(
    profile["lon"],
    profile["lat"],
    c=mean_t_by_station.reindex(profile.index),
    cmap="coolwarm",
    s=18,
)
axes[1].set_title("Station network - colored by daily mean t (degC)")
axes[1].set_xlabel("lon")
axes[1].set_ylabel("lat")
plt.colorbar(axes[1].collections[0], ax=axes[1])
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/08_spatial_network.png", dpi=130)
plt.close()

# --- altitude vs temperature: lapse rate check ---
fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(
    profile["height_sta"], mean_t_by_station.reindex(profile.index), s=12, alpha=0.6
)
ax.set_xlabel("height_sta (m)")
ax.set_ylabel("daily mean t (degC)")
ax.set_title("Altitude vs mean temperature (expected: decreasing)")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/09_altitude_vs_temp.png", dpi=130)
valid = mean_t_by_station.reindex(profile.index).notna() & profile["height_sta"].notna()
corr_alt_t = np.corrcoef(
    profile.loc[valid, "height_sta"], mean_t_by_station.reindex(profile.index)[valid]
)[0, 1]
print("Correlation altitude vs mean daily temperature:", round(corr_alt_t, 3))
plt.close()

# --- daily cycle: sample a handful of stations with full coverage across altitude range ---
full_cov = profile[(profile["n_timestamps"] == 240) & profile["has_temp"]]
sample_ids = (
    full_cov.sort_values("height_sta")
    .iloc[[0, len(full_cov) // 3, 2 * len(full_cov) // 3, -1]]
    .index
)

fig, ax = plt.subplots(figsize=(10, 5))
for sid in sample_ids:
    sub = df[df["number_sta"] == sid].sort_values("date")
    ax.plot(
        sub["date"],
        sub["t"] - 273.15,
        label=f"station {sid} ({profile.loc[sid,'height_sta']:.0f}m)",
    )
ax.set_ylabel("t (degC)")
ax.set_title("Temperature daily cycle - stations across the altitude range")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/10_daily_cycle_temp.png", dpi=130)
plt.close()

print("Sampled stations for daily cycle:", list(sample_ids))
print("Saved figures to", FIG_DIR)
