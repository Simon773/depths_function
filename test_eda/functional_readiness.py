import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from data_utils import load_raw

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR = str(RESULTS_DIR)

df = load_raw()

for var in ["t", "hu", "ff", "precip"]:
    pivot = df.pivot_table(index="number_sta", columns="date", values=var)
    n_full = (pivot.notna().all(axis=1)).sum()
    coverage = pivot.notna().mean().mean()
    print(
        f"{var}: {pivot.shape[0]} stations x {pivot.shape[1]} timestamps | "
        f"{n_full} stations fully complete | avg cell coverage {coverage:.1%}"
    )

# temperature is the best candidate: build the matrix and show a completeness heatmap
pivot_t = df.pivot_table(index="number_sta", columns="date", values="t")
fig, ax = plt.subplots(figsize=(10, 6))
ax.imshow(pivot_t.notna().values, aspect="auto", cmap="Greys", interpolation="none")
ax.set_xlabel("time index (6-min steps over the day)")
ax.set_ylabel("station index")
ax.set_title(
    "Temperature availability matrix (black = observed) - basis for functional depth"
)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/13_functional_matrix_availability.png", dpi=130)
plt.close()

pivot_t.to_csv(f"{FIG_DIR}/temperature_station_by_time_matrix.csv")
print(
    f"\nSaved temperature station-by-time matrix to {FIG_DIR}/temperature_station_by_time_matrix.csv"
)
print(
    "Fully complete stations for t:",
    (pivot_t.notna().all(axis=1)).sum(),
    "/",
    pivot_t.shape[0],
)
