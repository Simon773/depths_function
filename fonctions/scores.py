"""
Canonical score orientation, rank normalisation and cross-method overlap.

Every detector here is converted to a single convention:
    higher value = more anomalous ("outlyingness").
Depth functions are the exception in the raw code: a LOW depth means anomalous,
so they are negated once, here, and nowhere else.
"""

import numpy as np
import pandas as pd

# True  -> raw score already means "higher = more anomalous"
# False -> raw score means "lower = more anomalous" (depth functions)
HIGH_IS_ANOMALOUS = {
    "HBOS": True,
    "ABOD": True,
    "LOF": True,
    "DBSCAN": True,
    "CBLOF": True,
    "LDOF": True,
    "INFLO": True,
    "ODC": True,
    "OF": True,
    "CLOPD": True,
    "ROCF": True,
    "halfspace_depth": False,
    "simplicial_depth": False,
    "projection_depth": False,
}


def is_high_anomalous(method):
    """Case-insensitive lookup: the pipeline writes HALFSPACE_DEPTH,
    the registry declares halfspace_depth."""
    key = str(method).strip().lower()
    table = {k.lower(): v for k, v in HIGH_IS_ANOMALOUS.items()}
    if key not in table:
        raise KeyError(f"Unknown method {method!r}; add it to HIGH_IS_ANOMALOUS.")
    return table[key]


def to_outlyingness(scores, method):
    """Return scores on the common 'higher = more anomalous' scale."""
    scores = np.asarray(scores, dtype=float)
    return scores if is_high_anomalous(method) else -scores


def rank_normalise(scores):
    """Map scores to [0, 1] by rank. Scale-free, unlike min-max."""
    scores = np.asarray(scores, dtype=float)
    order = scores.argsort().argsort().astype(float)
    return order / max(len(scores) - 1, 1)


def minmax_normalise(scores):
    scores = np.asarray(scores, dtype=float)
    lo, hi = np.nanmin(scores), np.nanmax(scores)
    return np.zeros_like(scores) if hi == lo else (scores - lo) / (hi - lo)


def outlier_flags(scores_df, thresholds):
    """
    Boolean outlier matrix, one column per method.

    scores_df  : raw scores, one column per method (as produced by the pipeline)
    thresholds : {method: threshold} expressed on the RAW scale, so the values
                 already reported in the article stay usable as-is.
    """
    flags = {}
    for method, thr in thresholds.items():
        raw = np.asarray(scores_df[method], dtype=float)
        flags[method] = raw > thr if is_high_anomalous(method) else raw < thr
    return pd.DataFrame(flags, index=scores_df.index)


def detection_rates(flags_df):
    return flags_df.mean().sort_values(ascending=False)


def jaccard_matrix(flags_df):
    """Pairwise Jaccard index between the outlier sets of each method."""
    methods = list(flags_df.columns)
    out = pd.DataFrame(index=methods, columns=methods, dtype=float)
    for a in methods:
        for b in methods:
            sa, sb = flags_df[a].values, flags_df[b].values
            union = (sa | sb).sum()
            out.loc[a, b] = 1.0 if union == 0 else (sa & sb).sum() / union
    return out


def consensus(flags_df):
    """Number of methods flagging each observation, descending."""
    return flags_df.sum(axis=1).sort_values(ascending=False)


def spearman_matrix(scores_df):
    """
    Rank correlation between methods on the common outlyingness scale.
    Threshold-free, so it separates 'the methods disagree' from
    'the thresholds were picked differently'.
    """
    oriented = pd.DataFrame(
        {m: to_outlyingness(scores_df[m], m) for m in scores_df.columns},
        index=scores_df.index,
    )
    return oriented.corr(method="spearman")
