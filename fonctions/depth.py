"""
Fonction pour calculer les depths functions et leur score
"""

from depth.model.DepthEucl import DepthEucl
import fonctions.plot as p

def halfspace(X):
    model=DepthEucl().load_dataset(X)
    depthX=model.halfspace(X,exact=False,NRandom=3000,solver='simplerandom',output_option='lowest_depth')

    # We evaluate the dataset itself (calculate the depth of X with respect to X)
    # exact=True is fast here because we are in 2D with few points
    depth_scores_halfspace = model.halfspace(evaluate_dataset=True, exact=False,NRandom=3000, solver='simplerandom',output_option='lowest_depth')
    return(depth_scores_halfspace)

def simplicial(X):
    model=DepthEucl().load_dataset(X)
    depth_scores_simplicial=model.simplicial(X,exact=False,k=4000,evaluate_dataset=True)
    return(depth_scores_simplicial)

def projection(X):
    model=DepthEucl().load_dataset(X)

    # We evaluate the dataset itself (calculate the depth of X with respect to X)
    # exact=True is fast here because we are in 2D with few points
    depth_scores_projection = model.projection(X,evaluate_dataset=True,NRandom=10000,output_option="lowest_depth")
    return(depth_scores_projection)

def calcul_depth(X, choice_depth):
    if choice_depth == "halfspace_depth":
        scores = halfspace(X)
    elif choice_depth == "simplicial_depth":
        scores = simplicial(X)
    elif choice_depth == "projection_depth":
        scores = projection(X)
    p.plot_thresholds(scores, choice_depth)
    p.plot_thresholds_zoom(scores, choice_depth)
    return scores
