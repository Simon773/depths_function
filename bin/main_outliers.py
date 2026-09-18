# Import des libraries
import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path
import time

racine_projet = Path(__file__).resolve().parent.parent
if str(racine_projet) not in sys.path:
    sys.path.append(str(racine_projet))
import fonctions.depth as calc_depth
import fonctions.plot as p
import fonctions.algo_stat as calc_stat

# Paramètres
outlier_method = "CLOPD" #depth_function HBOS ABOD CBLOF
choice_depth = "halfspace_depth"

# Import des données 
data = pd.read_csv(os.path.join(racine_projet, "data/data_station_selected.csv"), parse_dates=["date"])
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
X = np.array(data[["dd","ff","precip","hu","t"]])

# type: "score" = continu, on seuille par quantile
#       "binary" = déjà 0/1 (ou -1/1), on garde tel quel
methods = {
    "HBOS":   {"func": calc_stat.calcul_HBOS,   "type": "score"},
    "ABOD":   {"func": calc_stat.calcul_ABOD,   "type": "score"},
    "LDOF":   {"func": calc_stat.calcul_LDOF,   "type": "score"},
    "LOF":    {"func": calc_stat.calcul_LOF,    "type": "score"},
    "INFLO":  {"func": calc_stat.calcul_INFLO,  "type": "score"},
    "DBSCAN": {"func": calc_stat.calcul_DBSCAN, "type": "score"},  # decision_function -> continu malgré le nom
    "ODC":    {"func": calc_stat.calcul_ODC,    "type": "score"},
    "OF":     {"func": calc_stat.calcul_OF,     "type": "score"},
    "CLOPD":  {"func": calc_stat.calcul_CLOPD,  "type": "score"},
    "ROCF":   {"func": calc_stat.calcul_ROCF,   "type": "score"},
    "CBLOF":  {"func": calc_stat.calcul_CBLOF,  "type": "score"},
    "HALFSPACE_DEPTH" : {"func": calc_depth.halfspace, "type":"score"},
    "PROJECTION_DEPTH" : {"func": calc_depth.projection, "type":"score"},
    "SIMPLICIAL_DEPTH" : {"func": calc_depth.simplicial, "type":"score"}
}

contamination = 0.02

raw_scores = pd.DataFrame(index=data.index)
binary_flags = pd.DataFrame(index=data.index)
timings = pd.DataFrame(index=data.index)

for name, info in methods.items():
    print(f"------- Calcul : {name}")
    try:
        start = time.perf_counter()
        result = info["func"](X)
        elapsed = time.perf_counter() - start
        print(f"  -> Temps : {elapsed:.3f} s")
        raw_scores[name] = result

        if info["type"] == "score":
            # seuil = quantile (1 - contamination) des scores
            seuil = np.quantile(result, 1 - contamination)
            p.plot_scores(result, seuil, name)
            binary_flags[name] = (result >= seuil).astype(int)
        else:
            # déjà binaire (ex: -1/1 pour DBSCAN natif, ou 0/1)
            binary_flags[name] = (np.array(result) == -1).astype(int) \
                if set(np.unique(result)) <= {-1, 1} \
                else np.array(result).astype(int)
            p.plot_scores(result, 0.5, name)
        timings[name] = elapsed

    except Exception as e:
        print(f"  -> ÉCHEC {name}: {e}")
        raw_scores[name] = np.nan
        binary_flags[name] = np.nan
        timings[name] = np.nan

# Score de consensus : combien de méthodes flaguent ce point comme outlier
binary_flags["consensus"] = binary_flags.sum(axis=1)

# Export
final = pd.concat([data[["dd","ff","precip","hu","t"]], binary_flags], axis=1)
final.to_csv("comparaison_outliers_binaire.csv", index=False, sep=";")

raw_scores.to_csv("resultats/comparaison_outliers_scores_bruts.csv", index=False, sep=";")

print("\n------ Points flagués par au moins la moitié des méthodes")
n_methods = len(methods)
print(final[final["consensus"] >= n_methods / 2])