# Import des libraries
import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt

racine_projet = Path(__file__).resolve().parent.parent


# Import des données 
data = pd.read_csv(os.path.join(racine_projet, "data/SE_20160101.csv"), parse_dates=["date"])
data = data[["number_sta", "lat", "lon", "date", "dd", "ff", "precip", "hu", "t"]]
# On laisse de côté psl car trop de données manquantes
print(f"Avant dropna : {len(data)}")
data = data.dropna(subset = ["dd", "ff", "precip", "hu", "t"])
print(f"Après dropna : {len(data)}")
data["heure"] = data["date"].dt.hour
data["minute"] = data["date"].dt.minute
data["seconde"] = data["date"].dt.second
data["heure_complete"] = pd.to_timedelta(
    data["heure"].astype(str) + ":" +
    data["minute"].astype(str) + ":" +
    data["seconde"].astype(str)
)
print("------- Import réussi")

# Préparation X
# data = data[data["number_sta"] == 1089001].reset_index() #temporaire
cols = ["dd", "ff", "precip", "hu", "t"]
X = np.array(data[cols])
X_df = pd.DataFrame(X, columns=cols)

plt.figure(figsize=(8, 6))
sns.boxplot(data=X_df)
plt.title("Boxplot")
plt.ylabel("Values")
plt.show()

# Sélection station
from scipy.stats import wasserstein_distance
import itertools
import random

cols = ["dd", "ff", "precip", "hu", "t"]
print(data.columns)
stations = data["number_sta"].unique()
n_selected = 10

def score(subset_stations):
    """Somme des distances de Wasserstein sur chaque variable."""
    d_sel = data[data["number_sta"].isin(subset_stations)]
    total = 0
    for c in cols:
        total += wasserstein_distance(data[c].dropna(), d_sel[c].dropna())
    return total

# Recherche gloutonne : on part d'un tirage aléatoire, 
# puis on améliore itérativement par échanges de stations
best = list(np.random.choice(stations, size=n_selected, replace=False))
best_score = score(best)

n_iter = 500
for _ in range(n_iter):
    candidate = best.copy()
    # on retire une station au hasard et on en ajoute une autre
    out_station = random.choice(candidate)
    in_station = random.choice([s for s in stations if s not in candidate])
    candidate.remove(out_station)
    candidate.append(in_station)
    
    s = score(candidate)
    if s < best_score:
        best, best_score = candidate, s

data_selected = data[data["number_sta"].isin(best)]
print("Score final :", best_score)

import matplotlib.pyplot as plt
import seaborn as sns

fig, axes = plt.subplots(1, len(cols), figsize=(4*len(cols), 5))

for i, c in enumerate(cols):
    df_plot = pd.concat([
        data[[c]].assign(source="Complet"),
        data_selected[[c]].assign(source="Sélection")
    ])
    sns.boxplot(data=df_plot, x="source", y=c, ax=axes[i])
    axes[i].set_title(c)

plt.tight_layout()
plt.show()

data_selected = data_selected[data.columns]  # réordonne/filtre les colonnes comme dans data
data_selected.to_csv("data/data_station_selected.csv", index=False)