# import helpers
import copy

import numpy as np
np.random.seed(43)
# import Dijkstra's shortest path algorithm
from scipy.sparse import csgraph, csr_matrix

# import graph building methods
from sklearn.neighbors import kneighbors_graph, radius_neighbors_graph


def graph_search(
    data,
    index,  # index of the factual point
    keys_immutable,
    model,
    n_neighbors=50,
    p_norm=2,
    mode="knn",
    frac=0.4,
    radius=0.001,
):
    # This one implements the FACE method from
    # Rafael Poyiadzi et al (2020), "FACE: Feasible and Actionable Counterfactual Explanations",
    # Conference on AI, Ethics & Accountability (AIES, 2020)
    """
    :param data: df
    :param n_neighbors: int > 0; number of neighbors when constructing knn graph
    :param step: float > 0; step_size for growing spheres
    :param mode: str; either 'knn' or 'epsilon'
    :param model: classification model (either tf keras, pytorch or sklearn)
    :param p_norm: float=>1; denotes the norm (classical: 1 or 2)
    :param frac: float 0 < number =< 1; fraction of data for which we compute the graph; if frac = 1, and data set large, then compute long
    :param keys_immutable: list; list of input names that may not be searched over
    :param radius: float > 0; parameter for epsilon density graph
    :return: candidate_counterfactual_star: np array (min. cost counterfactual explanation)
    """
    # Choose a subset of data for computational efficiency
    data = choose_random_subset(data, frac, index)

    # ADD CONSTRAINTS by immutable inputs into adjacency matrix
    # if element in adjacency matrix 0, then it cannot be reached
    # this ensures that paths only take same sex / same race / ... etc. routes
    
    if len(keys_immutable) != 0:
        for i in range(len(keys_immutable)):
            immutable_constraint_matrix1, immutable_constraint_matrix2 = build_constraints(
                data, i, keys_immutable
            )
    # There is no immutable variables
    # Therefore, all elements in the adjacency matrix are 1 (can be reached)
    else:
        num_row = data.shape[0]
        immutable_constraint_matrix1 = np.zeros([num_row, num_row], dtype = float)
        immutable_constraint_matrix2 = copy.deepcopy(immutable_constraint_matrix1)

    # POSITIVE PREDICTIONS
    y_predicted = model.predict_proba(data.values)
    y_predicted = np.argmax(y_predicted, axis=1)
    y_positive_indeces = np.where(y_predicted == 1)

    if mode == "knn":
        boundary = 3  # chosen in ad-hoc fashion
        median = n_neighbors
        is_knn = True

    elif mode == "epsilon":
        boundary = 0.00  # chosen in ad-hoc fashion
        median = radius
        is_knn = False
    else:
        raise ValueError("Only possible values for mode are knn and epsilon")

    neighbors_list = [
        median - boundary,
        median,
        median + boundary,
    ]
    print(neighbors_list)

    # obtain candidate targets (CT); two conditions need to be met:
    # (1) CT needs to be predicted positively & (2) CT needs to have certain "density"
    # for knn I interpret 'certain density' as sufficient number of neighbours
    candidate_counterfactuals = []
    candidate_counterfactuals_paths = []
    candidate_counterfactuals_indeces = [[0] for i in range(len(neighbors_list))]

    for n in neighbors_list:
        neighbor_candidates, predecessors, neighbor_cf_indeces = find_counterfactuals(
            candidate_counterfactuals,
            data,
            immutable_constraint_matrix1,
            immutable_constraint_matrix2,
            index,
            n,
            y_positive_indeces,
            is_knn=is_knn,
        )
        candidate_counterfactuals += neighbor_candidates
        candidate_counterfactuals_paths.append(predecessors)
        candidate_counterfactuals_indeces[neighbors_list.index(n)] = neighbor_cf_indeces

    # A list of CF points [[1 x num_feature], [...]]
    candidate_counterfactual_star = np.array(candidate_counterfactuals)
    print("candidate counterfactuals are: {}".format(candidate_counterfactual_star.shape))
    print("predecessors shape: {}".format(np.shape(candidate_counterfactuals_paths)))
    # print("candidate indeces: {}".format(candidate_counterfactuals_indeces))

    # STEP 4 -- COMPUTE DISTANCES between x^F and candidate x^CF; else return NaN
    # 1) no counterfactual points found
    if candidate_counterfactual_star.size == 0:
        candidate_counterfactual_star = np.empty(
            data.values.shape[1],
        )
        candidate_counterfactual_star[:] = np.nan

        return candidate_counterfactual_star

    # 2) CF points found -> calculate the distance between F and CF.
    if p_norm == 1:
        c_dist = np.abs((data.values[index] - candidate_counterfactual_star)).sum(
            axis=1
        )
    elif p_norm == 2:
        c_dist = np.square((data.values[index] - candidate_counterfactuals)).sum(axis=1)
    else:
        raise ValueError("Distance not defined yet. Choose p_norm to be 1 or 2")

    # return the single CF instance with the shortest distance    
    min_index = np.argmin(c_dist)
    candidate_counterfactual_star = candidate_counterfactual_star[min_index]

    # return the k=4 shortest points 
    # min_index_set = np.argsort(c_dist)[:200]
    # candidate_counterfactual_star = candidate_counterfactual_star[min_index_set]

    # hops_collection = retrieve_shortest_path(min_index, candidate_counterfactuals_indeces, candidate_counterfactuals_paths)

    return candidate_counterfactual_star


def retrieve_shortest_path(min_index, candidate_cf_indeces, candidate_cf_paths):
    # get all the nodes along the path
    graph_path = -1
    node_index = -1
    indeces_lower = len(candidate_cf_indeces[0])
    indeces_medium = len(candidate_cf_indeces[1])
    indeces_upper = len(candidate_cf_indeces[2])

    if min_index < indeces_lower:
        graph_path = candidate_cf_paths[0]
        node_index = candidate_cf_indeces[0][min_index]
    elif min_index < indeces_lower*2 + indeces_medium:
        graph_path = candidate_cf_paths[1]
        node_index = candidate_cf_indeces[1][min_index - indeces_lower*2]
    else:
        graph_path = candidate_cf_paths[2]
        node_index = candidate_cf_indeces[2][min_index - indeces_lower*4 - indeces_medium*2]
    
    graph_path = candidate_cf_paths[2]

    hop_idx = node_index
    hops_collection = [hop_idx, graph_path[hop_idx]]  

    while graph_path[hop_idx] != 0:
        hop_idx = graph_path[hop_idx]
        hops_collection.append(graph_path[hop_idx])
    
    print(hops_collection)

    return hops_collection


def choose_random_subset(data, frac, index):
    """
    Choose a subset of data for computational efficiency

    Parameters
    ----------
    data : pd.DataFrame
    frac: float 0 < number =< 1
        fraction of data for which we compute the graph; if frac = 1, and data set large, then compute long
    index: int

    Returns
    -------
    pd.DataFrame
    """
    number_samples = np.int(np.rint(frac * data.values.shape[0]))
    list_to_choose = (
        np.arange(0, index).tolist()
        + np.arange(index + 1, data.values.shape[0]).tolist()
    )
    chosen_indeces = np.random.choice(
        list_to_choose,
        replace=False,
        size=number_samples,
    )
    chosen_indeces = [
        index
    ] + chosen_indeces.tolist()  # make sure sample under consideration included
    data = data.iloc[chosen_indeces]
    data = data.sort_index()
    return data


def build_constraints(data, i, keys_immutable, epsilon=0.5):
    """

    Parameters
    ----------
    data: pd.DataFrame
    i : int
        Position of immutable key
    keys_immutable: list[str]
        Immutable feature
    epsilon: int

    Returns
    -------
    np.ndarray, np.ndarray
    """
    immutable_constraint_matrix = np.outer(
        data[keys_immutable[i]].values + epsilon,
        data[keys_immutable[i]].values + epsilon,
    )
    immutable_constraint_matrix1 = immutable_constraint_matrix / ((1 + epsilon) ** 2)
    immutable_constraint_matrix1 = ((immutable_constraint_matrix1 == 1) * 1).astype(
        float
    )
    immutable_constraint_matrix2 = immutable_constraint_matrix / (epsilon**2)
    immutable_constraint_matrix2 = ((immutable_constraint_matrix2 == 1) * 1).astype(
        float
    )
    return immutable_constraint_matrix1, immutable_constraint_matrix2


def find_counterfactuals(
    candidates,
    data,
    immutable_constraint_matrix1,
    immutable_constraint_matrix2,
    index,
    n,  # Lower range of neighbour list - medium - upper range
    y_positive_indeces,
    is_knn,
):
    """
    Steps 1 to 3 of the FACE algorithm

    Parameters
    ----------
    candidate_counterfactuals_star: list
    data: pd.DataFrame
    immutable_constraint_matrix1: np.ndarray
    immutable_constraint_matrix2: np.ndarray
    index: int
    n: int
    y_positive_indeces: int
    is_knn: bool

    Returns
    -------
    list
    """
    candidate_counterfactuals_star = copy.deepcopy(candidates)
    # STEP 1 -- BUILD NETWORK GRAPH
    graph = build_graph(
        data, immutable_constraint_matrix1, immutable_constraint_matrix2, is_knn, n
    )
    # STEP 2 -- APPLY SHORTEST PATH ALGORITHM  ## indeces=index (corresponds to x^F)
    distances, min_distance, predecessors = shortest_path(graph, index)
    # STEP 3 -- FIND COUNTERFACTUALS
    # minimum distance candidate counterfactuals
    candidate_min_distances = [
        min_distance,
        min_distance + 1,
        min_distance + 2,
        min_distance + 3,
    ]
    min_distance_indeces = np.array([0])
    for min_dist in candidate_min_distances:
        min_distance_indeces = np.c_[
            min_distance_indeces, np.array(np.where(distances == min_dist))
        ]
    min_distance_indeces = np.delete(min_distance_indeces, 0)
    indeces_counterfactuals = np.intersect1d(
        np.array(y_positive_indeces), np.array(min_distance_indeces)
    )
    for i in range(indeces_counterfactuals.shape[0]):
        candidate_counterfactuals_star.append(data.values[indeces_counterfactuals[i]])

    return candidate_counterfactuals_star, predecessors, indeces_counterfactuals


def shortest_path(graph, index):
    """
    Uses dijkstras shortest path

    Parameters
    ----------
    graph: CSR matrix
    index: int

    Returns
    -------
    np.ndarray, float, np.ndarray
    """
    # indices: only compute the path from the given point
    distances, predecessors = csgraph.dijkstra(
        csgraph=graph, directed=False, indices=index, return_predecessors=True
    )
    distances[index] = np.inf  # avoid min. distance to be x^F itself
    min_distance = distances.min()
    return distances, min_distance, predecessors


def build_graph(
    data, immutable_constraint_matrix1, immutable_constraint_matrix2, is_knn, n
):
    """

    Parameters
    ----------
    data: pd.DataFrame
    immutable_constraint_matrix1: np.ndarray
    immutable_constraint_matrix2: np.ndarray
    is_knn: bool
    n: int

    Returns
    -------
    CSR matrix
    """
    if is_knn:
        graph = kneighbors_graph(data.values, n_neighbors=n, n_jobs=-1)
    else:
        graph = radius_neighbors_graph(data.values, radius=n, n_jobs=-1)
    adjacency_matrix = graph.toarray()
    adjacency_matrix = np.multiply(
        adjacency_matrix,
        immutable_constraint_matrix1,
        immutable_constraint_matrix2,
    )  # element wise multiplication
    graph = csr_matrix(adjacency_matrix)
    return graph
