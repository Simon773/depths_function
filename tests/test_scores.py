"""
Smoke tests for the orientation logic. No data and no depth package needed.
Run from anywhere:  python tests/test_scores.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fonctions.scores as sc

rng = np.random.default_rng(0)
N = 2400


def check(label, condition):
    print(f"[{'OK  ' if condition else 'FAIL'}] {label}")
    return condition


def main():
    ok = True

    # 1. Every method the pipeline can produce must be declared.
    expected = {
        "HBOS",
        "ABOD",
        "LOF",
        "DBSCAN",
        "CBLOF",
        "LDOF",
        "INFLO",
        "ODC",
        "OF",
        "CLOPD",
        "ROCF",
        "halfspace_depth",
        "simplicial_depth",
        "projection_depth",
    }
    ok &= check("registry covers every method", expected == set(sc.HIGH_IS_ANOMALOUS))

    # 2. An unknown method must raise, not silently pass through.
    try:
        sc.to_outlyingness([1.0, 2.0], "not_a_method")
        ok &= check("unknown method raises", False)
    except KeyError:
        ok &= check("unknown method raises", True)

    # 3. Depths get flipped, classical scores do not.
    raw = np.array([0.1, 0.9])
    ok &= check(
        "depth is negated",
        np.allclose(sc.to_outlyingness(raw, "halfspace_depth"), -raw),
    )
    ok &= check(
        "classical score is untouched",
        np.allclose(sc.to_outlyingness(raw, "HBOS"), raw),
    )

    # 4. Two detectors agreeing on the same anomalies must correlate POSITIVELY
    #    once oriented, even though one is a depth and the other is not.
    truth = rng.random(N) < 0.05
    depth = np.where(truth, rng.uniform(0, 0.002, N), rng.uniform(0.01, 0.3, N))
    hbos = np.where(truth, rng.uniform(8, 15, N), rng.uniform(0, 5, N))
    df = pd.DataFrame({"halfspace_depth": depth, "HBOS": hbos})

    rho = sc.spearman_matrix(df).loc["halfspace_depth", "HBOS"]
    ok &= check(f"oriented spearman is positive (rho={rho:.3f})", rho > 0)

    # 5. Flags use the RAW scale, so article thresholds stay usable as written.
    flags = sc.outlier_flags(df, {"halfspace_depth": 0.005, "HBOS": 7.0})
    jac = sc.jaccard_matrix(flags).loc["halfspace_depth", "HBOS"]
    ok &= check(f"agreeing detectors give jaccard ~1 (got {jac:.3f})", jac > 0.95)
    ok &= check(
        "consensus max equals number of methods", sc.consensus(flags).max() == 2
    )

    # 6. Rank normalisation is immune to a single extreme value; min-max is not.
    #    This is the Figure 9 artefact.
    spiky = np.concatenate([rng.uniform(0, 0.01, N - 1), [50.0]])
    mm = np.sort(sc.minmax_normalise(spiky))[N // 2]
    rk = np.sort(sc.rank_normalise(spiky))[N // 2]
    ok &= check(f"min-max is crushed by one outlier (median={mm:.4f})", mm < 0.01)
    ok &= check(f"rank normalisation is not (median={rk:.4f})", 0.45 < rk < 0.55)

    print("\n" + ("ALL TESTS PASSED" if ok else "SOME TESTS FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
