"""
Sanity check on the real pipeline output.

    python tests/check_resultats.py resultats/comparaison_outliers_scores_bruts.csv

Catches the two failure modes that do not raise any exception:
  - a depth column already negated upstream (double negation)
  - a detection rate that disagrees with the threshold written in the article
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fonctions.scores as sc

# Thresholds as reported in the article, on the RAW scale.
# Adjust to whatever bin/comparaison_resultats actually uses.
ARTICLE_THRESHOLDS = {
    "HALFSPACE_DEPTH": 0.001,
    "SIMPLICIAL_DEPTH": 0.3,
    "PROJECTION_DEPTH": 0.08,
}

ARTICLE_COUNTS = {
    "HALFSPACE_DEPTH": 118,
    "SIMPLICIAL_DEPTH": 86,
    "PROJECTION_DEPTH": 5,
}


def check_orientation(df):
    """A raw depth lives in [0, 1]. Negative values mean the column was
    already flipped upstream, so scores.py would flip it a second time."""
    print("\n--- orientation ---")
    for col in df.columns:
        try:
            high_is_bad = sc.is_high_anomalous(col)
        except KeyError:
            print(f"  ?  {col}: not in the registry, add it to scores.py")
            continue
        n_nan = int(df[col].isna().sum())
        if n_nan == len(df):
            print(f"  !! {col}: entirely NaN -> this method produced nothing")
            continue
        if n_nan:
            print(f"  !  {col}: {n_nan} NaN values ({100*n_nan/len(df):.1f}%)")
        lo, hi = df[col].min(), df[col].max()
        if not high_is_bad and lo < 0:
            print(
                f"  !! {col}: range [{lo:.4g}, {hi:.4g}] is negative "
                f"-> already flipped upstream, remove that negation"
            )
        else:
            print(f"  ok {col}: range [{lo:.4g}, {hi:.4g}]")


def check_rates(df):
    """Recompute the counts the article reports, from the thresholds it states."""
    print("\n--- detection rates vs article ---")
    n = len(df)
    print(f"  n = {n}")
    for method, thr in ARTICLE_THRESHOLDS.items():
        if method not in df.columns:
            continue
        flags = sc.outlier_flags(df[[method]], {method: thr})[method]
        got = int(flags.sum())
        expected = ARTICLE_COUNTS.get(method)
        mark = "ok" if expected is not None and got == expected else "!!"
        print(
            f"  {mark} {method}: threshold {thr} -> {got} outliers "
            f"({100*got/n:.1f}%), article says {expected}"
        )


def check_agreement(df):
    def known_col(c):
        try:
            sc.is_high_anomalous(c)
            return df[c].notna().any()
        except KeyError:
            return False

    known = [c for c in df.columns if known_col(c)]
    rho = sc.spearman_matrix(df[known])
    print("\n--- pairwise rank correlation (threshold-free) ---")
    print(rho.round(2).to_string())
    print("\n  Strongly negative pairs would mean an orientation is still wrong:")
    pairs = rho.where(~np.eye(len(known), dtype=bool)).stack().sort_values()
    for (a, b), v in pairs.head(5).items():
        print(f"    {a} vs {b}: {v:.2f}")


def sweep_thresholds(
    df,
    method="HALFSPACE_DEPTH",
    grid=(0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.008, 0.01),
):
    """Which threshold reproduces the rate shown in Figure 8?
    Section 4 and Figure 8 disagree, so locate the value Figure 8 was built with."""
    if method not in df.columns:
        return
    print(f"\n--- threshold sweep on {method} ---")
    n = len(df)
    raw = df[method].to_numpy(dtype=float)
    high_is_bad = sc.is_high_anomalous(method)
    for thr in grid:
        count = int((raw > thr).sum() if high_is_bad else (raw < thr).sum())
        print(f"  threshold {thr:<8g} -> {count:5d} outliers  ({100*count/n:5.1f}%)")
    print("  Figure 8 of the article shows ~19.5% for this method.")


def main():
    path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "resultats/comparaison_outliers_scores_bruts.csv"
    )
    df = pd.read_csv(path, sep=";")
    df = df.select_dtypes(include=[np.number])
    print(f"Loaded {path}: {df.shape[0]} rows, {df.shape[1]} numeric columns")

    check_orientation(df)
    check_rates(df)
    check_agreement(df)
    sweep_thresholds(df)


if __name__ == "__main__":
    main()
