import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from pathlib import Path
racine_projet = Path(__file__).resolve().parent.parent
if str(racine_projet) not in sys.path:
    sys.path.append(str(racine_projet))
import fonctions.plot as p

raw_scores = pd.read_csv("resultats/comparaison_outliers_scores_bruts.csv", sep = ';')
print(raw_scores.columns)
raw_scores = raw_scores.drop(columns = ["INFLO"])

depth_cols = ['HALFSPACE_DEPTH','PROJECTION_DEPTH','SIMPLICIAL_DEPTH']
for c in depth_cols:
    raw_scores[c] = -raw_scores[c]  # inversion pour cohérence de sens

corr = raw_scores.drop(columns=['OF']).corr(method='spearman')
print(corr)

plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Correlation")
plt.show()

# Transformation en binaire
seuil = {
    "HBOS":  10,
    "ABOD":  8000,
    "LDOF":  2.917,
    "LOF":   1.964,
    #"INFLO": 1.0,
    "DBSCAN": 0.99, #A VERIFIER
    "ODC":  1.3, #A VERIFIER
    "CLOPD": 0.7447,
    "ROCF": 0.1,
    "CBLOF":  35.69,
    "HALFSPACE_DEPTH" : -0.005,
    "PROJECTION_DEPTH" : -0.08,
    "SIMPLICIAL_DEPTH" : -0.3
}

binary_scores = raw_scores.copy()

for col, s in seuil.items():
    if col in binary_scores.columns:
        binary_scores[col] = (binary_scores[col] >= s).astype(int)

import matplotlib.pyplot as plt
import numpy as np

n_cols = len(seuil)
n_rows = (n_cols + 2) // 3  # 3 graphiques par ligne

fig, axes = plt.subplots(n_rows, 3, figsize=(15, 3 * n_rows))
axes = axes.flatten()

for i, (col, s) in enumerate(seuil.items()):
    ax = axes[i]
    if col in raw_scores.columns:
        valeurs_triees = raw_scores[col].sort_values().reset_index(drop=True)
        ax.scatter(valeurs_triees.index, valeurs_triees.values, linewidth=1)
        ax.axhline(y=s, color='red', linestyle='--', label=f'seuil={s}')
        ax.set_title(col)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.legend(fontsize=8)
    else:
        ax.set_visible(False)

# Cacher les axes vides restants
for j in range(len(seuil), len(axes)):
    axes[j].set_visible(False)

plt.tight_layout()
plt.show()

taux_detection = binary_scores.mean().sort_values(ascending=False)
taux_detection.plot(kind='bar', figsize=(10, 5))
plt.ylabel("Proportion d'anomalies détectées")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

#### COmparaison sur un seul graphique
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(12, 7))

cmap = plt.get_cmap("tab20")
colors = [cmap(i) for i in range(len(seuil))]

for i, col in enumerate(seuil.keys()):
    if col in raw_scores.columns:
        valeurs = raw_scores[col].sort_values().reset_index(drop=True)
        valeurs_norm = (valeurs - valeurs.min()) / (valeurs.max() - valeurs.min())
        x_percentile = np.linspace(0, 100, len(valeurs_norm))
        ax.plot(x_percentile, valeurs_norm, label=col, linewidth=1.5, color=colors[i])

ax.set_xlabel("Percentile")
ax.set_ylabel("Normalised score (0-1)")
ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
plt.tight_layout()
plt.show()

#### COmparaison des queues de distributions
n_cols = len(seuil)
n_rows = (n_cols + 2) // 3

fig, axes = plt.subplots(n_rows, 3, figsize=(15, 4 * n_rows))
axes = axes.flatten()

for i, (col, s) in enumerate(seuil.items()):
    ax = axes[i]
    if col in raw_scores.columns:
        valeurs_triees = raw_scores[col].sort_values().reset_index(drop=True)
        n = len(valeurs_triees)
        debut = int(n * 0.9)  # ne garder que les 10% derniers points
        ax.plot(range(debut, n), valeurs_triees.iloc[debut:], linewidth=1.2, marker='o', markersize=3)
        ax.axhline(y=s, color='red', linestyle='--', label=f'seuil={s}')
        ax.set_title(col)
        ax.legend(fontsize=8)
    else:
        ax.set_visible(False)

for j in range(len(seuil), len(axes)):
    axes[j].set_visible(False)

plt.tight_layout()
plt.show()