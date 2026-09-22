"""
Fonction pour calculer les depths functions et leur score
"""

import numpy as np
from depth.model.DepthEucl import DepthEucl
from sklearn.preprocessing import StandardScaler

import fonctions.plot as p

# Same random-direction budget for every approximated depth, otherwise neither
# the scores nor the runtimes in the timing table are comparable.
NRANDOM = 3000


def _scale(X):
    """Depths are affine invariant in theory, but the Monte-Carlo
    approximations are not: psl (~1e5) would dominate every random direction."""
    return StandardScaler().fit_transform(np.asarray(X, dtype=float))


def halfspace(X):
    Xs = _scale(X)
    model = DepthEucl().load_dataset(Xs)
    return model.halfspace(
        evaluate_dataset=True,
        exact=False,
        NRandom=NRANDOM,
        solver="simplerandom",
        output_option="lowest_depth",
    )


def simplicial(X):
    Xs = _scale(X)
    model = DepthEucl().load_dataset(Xs)
    return model.simplicial(Xs, exact=False, k=NRANDOM, evaluate_dataset=True)


def projection(X):
    Xs = _scale(X)
    model = DepthEucl().load_dataset(Xs)
    return model.projection(
        Xs,
        evaluate_dataset=True,
        NRandom=NRANDOM,
        output_option="lowest_depth",
    )


def calcul_depth(X, choice_depth):
    if choice_depth == "halfspace_depth":
        scores = halfspace(X)
    elif choice_depth == "simplicial_depth":
        scores = simplicial(X)
    elif choice_depth == "projection_depth":
        scores = projection(X)
    else:
        raise ValueError(f"Unknown depth {choice_depth!r}")
    p.plot_thresholds(scores, choice_depth)
    p.plot_thresholds_zoom(scores, choice_depth)
    return scores
