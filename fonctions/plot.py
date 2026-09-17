import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc

def plot_thresholds(scores, title):
    # sorted scores
    scores_sorted = np.sort(scores)

    index = np.arange(len(scores_sorted))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # ELBOW plot
    ax1.plot(index, scores_sorted, 'b-o', markersize=4)
    ax1.set_title("Simplicial Volume : Elbow Method")
    ax1.set_xlabel("Index (sorted)")
    ax1.set_ylabel("Depth")
    ax1.grid(True, alpha=0.3)

    # GAP plot
    ax2.scatter(index, scores_sorted, c='red', marker='+', s=60)
    ax2.set_title("Simplicial Volume : Gap Method")
    ax2.set_xlabel("Index (sorted)")
    ax2.set_ylabel("Depth")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(f"resultats/{title}.png")

def plot_thresholds_zoom(scores, title):
    # sorted scores
    scores_sorted = np.sort(scores)

    # Zoom on the lowest 40% to better see anomalies
    n_zoom = int(len(scores) * 0.40)
    scores_zoom = scores_sorted[:n_zoom]
    print(scores_zoom)
    index = np.arange(n_zoom)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # ELBOW plot
    ax1.plot(index, scores_zoom, 'b-o', markersize=4)
    ax1.set_title("halfspace depth : Elbow method (Zoom 40%)")
    ax1.set_xlabel("Index (sorted)")
    ax1.set_ylabel("Depth")
    ax1.grid(True, alpha=0.3)

    # GAP plot
    ax2.scatter(index, scores_zoom, c='red', marker='+', s=60)
    ax2.set_title("halfspace depth : Gap method (Zoom 40%)")
    ax2.set_xlabel("Index (sorted)")
    ax2.set_ylabel("Depth")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(f"resultats/{title}_zoom.png")

def plot_scores(scores, threshold, title):
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.scatter(np.arange(len(scores)), scores, c='blue', s=15, label = "Data")

    # Add threshold lines
    ax.axhline(y=threshold, color='green', linestyle='--', label=f'Threshold ({threshold})')

    ax.set_title("Sorted Depth Scores vs True Data")
    ax.set_xlabel("Index (sorted by depth)")
    ax.set_ylabel("Depth Score")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(f"resultats/plot_scores_{title}_{threshold}.png")

def confusion_matrix(y, y_pred, threshold):
    fig, ax = plt.subplots(figsize=(12, 6))

    cm_gap = confusion_matrix(y, y_pred)
    disp_gap = ConfusionMatrixDisplay(confusion_matrix=cm_gap, display_labels=['Normal', 'Anomaly'])
    disp_gap.plot(ax=ax, cmap='Blues', colorbar=False)
    ax.set_title(f"Confusion Matrix : (Threshold = {threshold})")

    plt.tight_layout()
    plt.show()

def plot_roc_curves(dict_scores, y_true):
    plt.figure(figsize=(8, 6))

    for name, scores in dict_scores.items():
        anomaly_scores = -np.array(np.sort(scores))
        fpr, tpr, thresholds = roc_curve(y_true, anomaly_scores)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.3f})', linestyle='--')

    plt.plot([0, 1], [0, 1], color='black', lw=1, linestyle='--')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('FP')
    plt.ylabel('TP')
    plt.title('Comparison of the Depth Functions')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.show()