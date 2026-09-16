import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from data_utils import NUMERIC_COLS, load_raw
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR = str(RESULTS_DIR)

df = load_raw()
vars_ = NUMERIC_COLS + ["height_sta"]

# --- linear (Pearson) and monotonic (Spearman) correlation ---
pearson = df[vars_].corr(method="pearson")
spearman = df[vars_].corr(method="spearman")

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
sns.heatmap(
    pearson, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=axes[0]
)
axes[0].set_title("Pearson correlation")
sns.heatmap(
    spearman, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=axes[1]
)
axes[1].set_title("Spearman correlation")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/11_correlation_pearson_spearman.png", dpi=130)
plt.close()


# --- PPS-equivalent: asymmetric, non-linear predictive score via a shallow decision tree ---
# Mirrors ppscore's approach: for each ordered pair (x -> y), fit a small tree,
# score with cross-validated R2 (regression case, all our vars are numeric),
# floor negative scores at 0 (worse than predicting the mean = no predictive power).
def pps_matrix(
    data: pd.DataFrame, cols: list[str], max_depth: int = 4, cv: int = 4
) -> pd.DataFrame:
    out = pd.DataFrame(index=cols, columns=cols, dtype=float)
    for x in cols:
        for y in cols:
            if x == y:
                out.loc[x, y] = 1.0
                continue
            sub = data[[x, y]].dropna()
            if len(sub) < 50:
                out.loc[x, y] = np.nan
                continue
            X = sub[[x]].values
            yv = sub[y].values
            model = DecisionTreeRegressor(max_depth=max_depth, random_state=0)
            scores = cross_val_score(model, X, yv, cv=cv, scoring="r2")
            out.loc[x, y] = max(scores.mean(), 0.0)
    return out.astype(float)


pps = pps_matrix(df, vars_)
pps.to_csv(f"{FIG_DIR}/pps_matrix.csv")

fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(pps, annot=True, fmt=".2f", cmap="viridis", vmin=0, vmax=1, ax=ax)
ax.set_title("PPS-equivalent (tree-based, asymmetric): row predicts column")
ax.set_xlabel("predicted (y)")
ax.set_ylabel("predictor (x)")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/12_pps_matrix.png", dpi=130)
plt.close()

print("Pearson matrix:\n", pearson.round(2))
print("\nPPS matrix:\n", pps.round(2))

# --- flag pairs where PPS and Pearson disagree a lot (non-linear signal Pearson misses) ---
diff = (pps - pearson.abs()).round(2)
print("\nPairs where PPS >> |Pearson| (possible non-linear relationship):")
flat = diff.where(~np.eye(len(vars_), dtype=bool)).stack().sort_values(ascending=False)
print(flat.head(10))
