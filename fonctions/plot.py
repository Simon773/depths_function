import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, auc, roc_curve
from sklearn.metrics import confusion_matrix as sk_confusion_matrix

import fonctions.scores as sc


def plot_thresholds(scores, title):
    scores_sorted = np.sort(scores)
    index = np.arange(len(scores_sorted))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(index, scores_sorted, "b-o", markersize=4)
    ax1.set_title("Elbow Method")
    ax1.set_xlabel("Index (sorted)")
    ax1.set_ylabel("Depth")
    ax1.grid(True, alpha=0.3)

    ax2.scatter(index, scores_sorted, c="red", marker="+", s=60)
    ax2.set_title("Gap Method")
    ax2.set_xlabel("Index (sorted)")
    ax2.set_ylabel("Depth")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(f"resultats/{title}.png")
    plt.close(fig)


def plot_thresholds_zoom(scores, title, zoom=0.40):
    scores_sorted = np.sort(scores)

    n_zoom = int(len(scores) * zoom)
    scores_zoom = scores_sorted[:n_zoom]
    index = np.arange(n_zoom)

    pct = int(zoom * 100)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(index, scores_zoom, "b-o", markersize=4)
    ax1.set_title(f"Elbow method (Zoom {pct}%)")
    ax1.set_xlabel("Index (sorted)")
    ax1.set_ylabel("Depth")
    ax1.grid(True, alpha=0.3)

    ax2.scatter(index, scores_zoom, c="red", marker="+", s=60)
    ax2.set_title(f"Gap method (Zoom {pct}%)")
    ax2.set_xlabel("Index (sorted)")
    ax2.set_ylabel("Depth")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(f"resultats/{title}_zoom.png")
    plt.close(fig)


def plot_scores(scores, threshold, title):
    scores_sorted = np.sort(scores)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.scatter(
        np.arange(len(scores_sorted)), scores_sorted, c="blue", s=15, label="Data"
    )
    ax.axhline(
        y=threshold, color="green", linestyle="--", label=f"Threshold ({threshold})"
    )

    ax.set_title("Sorted Depth Scores")
    ax.set_xlabel("Index (sorted by depth)")
    ax.set_ylabel("Depth Score")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(f"resultats/plot_scores_{title}_{threshold}.png")
    plt.close(fig)


def plot_confusion_matrix(y, y_pred, threshold):
    fig, ax = plt.subplots(figsize=(12, 6))

    cm = sk_confusion_matrix(y, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=["Normal", "Anomaly"]
    )
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion Matrix : (Threshold = {threshold})")

    plt.tight_layout()
    fig.savefig(f"resultats/confusion_matrix_{threshold}.png")
    plt.close(fig)


def plot_roc_curves(dict_scores, y_true, filename="roc_curves"):
    """
    Requires a real ground truth in y_true, aligned index by index with each
    score array. Scores are oriented but never sorted: sorting breaks the
    pairing with y_true and drives the AUC to 0.5.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    for name, scores in dict_scores.items():
        anomaly_scores = sc.to_outlyingness(scores, name)
        fpr, tpr, _ = roc_curve(y_true, anomaly_scores)
        ax.plot(
            fpr, tpr, lw=2, label=f"{name} (AUC = {auc(fpr, tpr):.3f})", linestyle="--"
        )

    ax.plot([0, 1], [0, 1], color="black", lw=1, linestyle="--")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Comparison of the Depth Functions")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.savefig(f"resultats/{filename}.png")
    plt.close(fig)


def plot_score_distributions(scores_df, filename="distribution_scores", use_ranks=True):
    """
    Figure 9 of the article. Every method is put on the common
    'higher = more anomalous' scale first, otherwise depth curves are mirrored
    relative to the others and the shapes are not comparable.

    use_ranks=True normalises by rank rather than min-max. Min-max is destroyed
    by a single extreme value, which is exactly the ROCF case, so the flat/steep
    contrast it produces is partly an artefact of the scaling.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    percentile = np.linspace(0, 100, len(scores_df))

    for method in scores_df.columns:
        oriented = sc.to_outlyingness(scores_df[method], method)
        norm = (
            sc.rank_normalise(oriented) if use_ranks else sc.minmax_normalise(oriented)
        )
        ax.plot(percentile, np.sort(norm), lw=1.5, label=method)

    ax.set_xlabel("Percentile")
    ax.set_ylabel("Normalised outlyingness (0-1)")
    ax.set_title("Distribution of normalised outlyingness scores")
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(f"resultats/{filename}.png", dpi=130)
    plt.close(fig)
