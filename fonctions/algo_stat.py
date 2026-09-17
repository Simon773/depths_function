import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.spatial.distance import pdist

from pyod.models.abod import ABOD
from pyod.models.hbos import HBOS
from pyod.models.lof import LOF
from pyod.models.hdbscan import HDBSCAN
from pyod.models.cblof import CBLOF
from sklearn.model_selection import train_test_split


def calcul_HBOS(X):
    X_train, X_test = train_test_split(X, random_state=42)
    clf = HBOS()
    clf.fit(X_train)
    return(clf.decision_function(X))

def calcul_ABOD(X):
    clf = ABOD(contamination = 0.2, method = 'fast')
    clf.fit(X)
    y_pred = clf.predict(X)
    return(-clf.decision_scores_)

def calcul_LOF(X):
    X_train, X_test = train_test_split(X, random_state=42)
    clf = LOF()
    clf.fit(X_train)
    return(clf.decision_function(X))

def calcul_DBSCAN(X):
    X_train, X_test = train_test_split(X, random_state=42)
    clf = HDBSCAN()
    clf.fit(X_train)
    return(clf.decision_function(X))

def calcul_CBLOF(X):
    X_train, X_test = train_test_split(X, random_state=42)
    clf = CBLOF()
    clf.fit(X_train)
    return(clf.decision_function(X))


def calcul_LDOF(X, k=5):
    """
    LDOF (Zhang et al.) : LDOF_k(xp) = d_xp / D_xp
    d_xp  = distance moyenne de xp à ses k plus proches voisins (KNN distance)
    D_xp  = distance moyenne entre les k plus proches voisins eux-mêmes (KNN inner distance)
    Un score élevé indique un point qui s'écarte fortement de son voisinage.
    """
    X = np.asarray(X)
    n = X.shape[0]
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(X)
    distances, indices = nbrs.kneighbors(X)
    distances = distances[:, 1:]   # on retire le point lui-même
    indices = indices[:, 1:]

    ldof_scores = np.zeros(n)
    for i in range(n):
        d_xp = distances[i].mean()
        neighbor_points = X[indices[i]]
        if len(neighbor_points) > 1:
            inner_dist = pdist(neighbor_points)
            D_xp = inner_dist.mean() if inner_dist.size > 0 else 1e-10
        else:
            D_xp = 1e-10
        ldof_scores[i] = d_xp / D_xp if D_xp > 0 else 0.0
    return ldof_scores


def calcul_INFLO(X, k=5):
    """
    INFLO (Jin et al.) : compare la densité d'un point à celle de son
    "influence space" IS(p) = kNN(p) ∪ RkNN(p) (voisins + voisins inverses).
    density(p) = 1 / distance moyenne à ses k plus proches voisins
    INFLO(p) = densité moyenne de IS(p) / densité(p)
    """
    X = np.asarray(X)
    n = X.shape[0]
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(X)
    distances, indices = nbrs.kneighbors(X)
    distances = distances[:, 1:]
    indices = indices[:, 1:]

    density = 1.0 / (distances.mean(axis=1) + 1e-10)

    reverse_knn = [set() for _ in range(n)]
    for i in range(n):
        for j in indices[i]:
            reverse_knn[j].add(i)

    inflo_scores = np.zeros(n)
    for i in range(n):
        influence_space = set(indices[i].tolist()) | reverse_knn[i]
        if not influence_space:
            inflo_scores[i] = 1.0
            continue
        avg_density_is = np.mean([density[o] for o in influence_space])
        inflo_scores[i] = avg_density_is / density[i]
    return inflo_scores


def calcul_ODC(X, k=3, p=1.5, max_iter=10):
    """
    ODC (Ahmed & Mahmood) : K-means modifié.
    Un point est suspecté outlier si sa distance à son centroïde dépasse
    p * (distance moyenne de tous les points à leur centroïde).
    On renvoie le ratio distance/(p*Md) comme score continu d'anomalie.
    """
    X = np.asarray(X)
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=max_iter)
    labels = kmeans.fit_predict(X)
    centroids = kmeans.cluster_centers_

    dist_to_centroid = np.linalg.norm(X - centroids[labels], axis=1)
    Md = dist_to_centroid.mean()
    scores = dist_to_centroid / (p * Md + 1e-10)
    return scores


def calcul_OF(X, rep=20, k=5, sample_ratio=0.5, random_state=42):
    """
    OF - Observability Factor (Ha et al.) :
    epsilon = moyenne des distances au k-ième voisin sur tout le data set.
    A chaque itération, on tire M échantillons aléatoires, on calcule leur
    voisinage epsilon (IN_eps), et chaque objet reçoit 1 s'il appartient à
    IN_eps, 0 sinon. OF(oi) = moyenne des delta_r sur les répétitions.
    Un score OF faible = suspect ; on inverse (1 - OF) pour rester cohérent
    avec les autres fonctions (score élevé = anomalie).
    """
    X = np.asarray(X)
    n = X.shape[0]

    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(X)
    kth_distances, _ = nbrs.kneighbors(X)
    epsilon = kth_distances[:, -1].mean()

    radius_nbrs = NearestNeighbors(radius=epsilon).fit(X)
    M = max(1, int(n * sample_ratio))
    rng = np.random.default_rng(random_state)

    delta_counts = np.zeros(n)
    for _ in range(rep):
        sample_idx = rng.choice(n, size=M, replace=False)
        in_eps_indices, _ = radius_nbrs.radius_neighbors(X[sample_idx])
        in_eps = set()
        for idx_list in in_eps_indices:
            in_eps.update(idx_list.tolist())
        for i in in_eps:
            delta_counts[i] += 1

    of_scores = delta_counts / rep
    return 1 - of_scores  # score élevé = anomalie


def calcul_CLOPD(X, k=5):
    """
    CLOPD (Anitha & Mary Metilda) :
    1) normalisation z-score des données
    2) distance euclidienne au k-ième plus proche voisin
    Les points ayant la plus grande distance à leur k-ième voisin sont
    les plus suspects (score = cette distance).
    """
    X = np.asarray(X)
    Z = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-10)

    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(Z)
    distances, _ = nbrs.kneighbors(Z)
    scores = distances[:, -1]  # distance au k-ième voisin (hors soi-même)
    return scores


def calcul_ROCF(X, n_clusters=5):
    """
    ROCF (Huang et al.) : au lieu de fixer un paramètre top-n, on considère
    que les petits clusters (par rapport à un cluster "normal") sont des
    clusters outliers. On approxime le MUNG par un clustering hiérarchique,
    on trie les clusters par taille décroissante C1 (le + grand) ... Cn,
    puis :
        ROCF(Ci) = 1 - exp( -|C_(i+1)| / |Ci|^2 )
    Chaque point reçoit le score ROCF de son cluster.
    """
    X = np.asarray(X)
    clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage='single').fit(X)
    labels = clustering.labels_

    unique, counts = np.unique(labels, return_counts=True)
    order = np.argsort(-counts)          # tri décroissant par taille
    sorted_sizes = counts[order]
    sorted_labels = unique[order]
    n_c = len(sorted_sizes)

    rocf_values = np.zeros(n_c)
    for i in range(n_c - 1):
        Ci = sorted_sizes[i]
        Ci1 = sorted_sizes[i + 1]
        rocf_values[i] = 1 - np.exp(-Ci1 / (Ci ** 2))
    if n_c > 1:
        rocf_values[-1] = rocf_values[-2]  # dernier cluster : même score que l'avant-dernier

    label_to_rocf = {sorted_labels[i]: rocf_values[i] for i in range(n_c)}
    scores = np.array([label_to_rocf[l] for l in labels])
    return scores