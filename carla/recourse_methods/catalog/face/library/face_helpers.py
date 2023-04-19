import numpy as np
import math
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity
from scipy.sparse import csr_matrix, csgraph

## ["epsilon", "knn", "kde"]
# epsilon-graph: the most basic method, as long as neighbours are within epsilon-distance, they are kept in the adjacency matrix. The weight of the edge is diminished (logarithmically, by default) by the distance (default is l2-norm).

# knn-graph: for all neighbours which are within the distance-threshold, only top-k nearest neighbours are kept in the adjacency matrix. Therefore, this method runs additional selection process. 

# kde-graph: for all points that are within the distance-threshold, the value of the edge (dist) is weighted inversely by the kernel density.

def get_weights_epsilon(n_samples, X, distance_threshold,                 
                  distance_function, weight_function):
    """Get the weight matrix in the epsilon graph

    Args:
        n_samples (int): number of rows in the dataframe
        X (pd.DataFrame): X
        distance_threshold (float): distance threshold
        distance_function (function(numpy.ndarray, numpy.ndarray) -> float):
                        calculate the distance between two points    
        weight_function (function(float) -> float): 
                        calculate the weight of an edge

    Returns:
        W: weight matrix of shape (n_samples x n_samples)
    """

    # Initialise the weight matrix
    W = np.zeros((n_samples, n_samples))

    # Calculate only the lower left of the weight matrix
    for i in range(n_samples):
        # get features of the ith data instance
        x_i = X.iloc[i].values.reshape(-1, 1)

        for j in range(n_samples):
            x_j = X.iloc[j].values.reshape(-1, 1)

            # In Line 2, Algorithm 1, there are 2 conditions to check
            # 1) distance between two points has to be smaller than or equal to distance_threshold.
            # 2) custom conditions c need to be satisfied.

            ## Condition 2: check custom conditions
            if not check_conditions(x_i, x_j):
                continue

            ## Condition 1: check distance
            dist = distance_function(x_i, x_j)
            if dist <= distance_threshold and dist != 0:
                W[i, j] = weight_function(dist)*dist
                # W[j, i] = W[i, j]
            
    return W


def get_weights_knn(n_samples, X, distance_threshold, n_neighbours, 
                    distance_function, weight_function):
    """Get the weight matrix in the knn graph

    Args:
        n_samples (int): number of rows in the dataframe
        X (Pd.DataFrame): X
        distance_threshold (float): distance threshold 
        n_neighbours (int): number of neighbours to be connected with
        distance_function (function(numpy.ndarray, numpy.ndarray) -> float):
                        calculate the distance between two points    
        weight_function (function(float) -> float): 
                        calculate the weight of an edge

    Returns:
        W: weight matrix of shape ((n_samples x n_samples)). 
    """
    W = np.zeros((n_samples, n_samples))
    # Step 1: find neighbours that are 1) within the distance threshold; 
    # 2) nearest k neighbours.    
    for i in range(n_samples):
        x_i = X.iloc[i].values.reshape(-1, 1)

        # Count the number of neighbours which are outside the distance-threshold
        counter = 0
        for j in range(n_samples):
            x_j = X.iloc[j].values.reshape(-1, 1)
            dist = distance_function(x_i, x_j)
            if (check_conditions(x_i, x_j) and dist <= distance_threshold):
                W[i, j] = dist
            else:
                counter += 1
        
        # Once we get distance between neighbours within the threshold,
        # we only keep the top-k nearest points, then reset the rest to 0
        #TODO: edge conditions need to be fixed, e.g. n is very large
        n_reset_neighbours = counter+n_neighbours+1
        reset_neighbours_idx = np.argsort(W[i, :])[n_reset_neighbours:]
        mask = np.ix_(reset_neighbours_idx)
        W[i, mask] = 0

    # Step 2: assign weight to the edge based on the distance between two vertices.
    for i in range(n_samples):
        x_i = X.iloc[i].values.reshape(-1, 1)
        for j in range(n_samples):
            x_j = X.iloc[j].values.reshape(-1, 1)
            if W[i, j] != 0:
                dist = W[i, j]
                W[i, j] = weight_function(dist)*dist
    
    return W

def get_weights_kde(n_samples, X, distance_threshold, 
                    distance_function, weight_function, density_scorer):
    """Get the weight matrix in the kde graph

    Args:
        n_samples (int): number of rows in the dataframe.
        X (pd.DataFrame): X
        distance_threshold (float): distance threshold
        distance_function (function(numpy.ndarray, numpy.ndarray) -> float):
                        Calculate the distance between two points.
        weight_function (function(float) -> float): 
                        Calculate the weight of an edge.
        density_scorer (function(array-like of shape(1, n_samples)) -> float): 
                        The function KernelDensity.score_samples() which returns the 
                        log-likelihood of this sample under the estimator.

    Returns:
        W: weight matrix of shape ((n_samples x n_samples)).
    """
    W = np.zeros((n_samples, n_samples))

    # Step 1: find neighbours within the distance threshold
    for i in range(n_samples):
        x_i = X.iloc[i].values.reshape(-1, 1)
        for j in range(n_samples):
            x_j = X.iloc[j].values.reshape(-1, 1)

            if not check_conditions(x_i, x_j):
                continue

            # The value of an edge is distance diminished by the density.
            dist = distance_function(x_i, x_j)
            if dist <= distance_threshold:
                 mid_points = (x_i + x_j)/2
                 # Density is the log-likelihood of the mid-point.
                 density = density_scorer(mid_points.reshape(1,-1))
                 W[i, j] = weight_function(np.exp(density)) * dist
    
    print(W[20, 218])
    return W

def get_kernel_density_estimator(X):
    """Get the best Kernel Density Estimator

    Args:
        X (pd.DataFrame): X

    Returns:
        kernel density estimator: the best fitted kernel density estimator 
    """
    # Assumption 1: the kernel to use is Gaussian
    # Assumption 2: use Euclidean distance (l2-norm) to calculate distance
    
    # bandwidth: the number of bins that divide the range of the data.
    bandwidths = np.logspace(-2, 0, 20)
    kd_estimators = GridSearchCV(
        estimator = KernelDensity(kernel="gaussian", metric="euclidean"),
        param_grid = {"bandwidth": bandwidths}
    )

    kd_estimators.fit(X)
    return kd_estimators.best_estimator_

def get_epsilon_knn_weight_function(k, n_features, n_samples):
    """Get the weight function to calculate weight of edge in knn/epsilon graph.

    Args:
        k (int): how significant you want for diminishing the distance between two points.
        n_features (int): number of features
        n_samples (int): number of data instances

    Returns:
        function: a weight function
    """
    # The volume of a d-dimensional unit sphere is pi**(d/2)/gamma(d/2+1)
    sphere_volume = math.pi**(n_features/2)/math.gamma(n_features/2 + 1)
    r = (k/(n_samples * sphere_volume))

    weight_function = lambda x: -np.log((r/x)**n_features)
    return weight_function


def get_weight_matrix(params):
    """Return the correct weight matrix.

    Returns:
        weight_matrix: weight matrix of shape ((n_samples, n_samples))
    """
    weight_matrix = 0
    if params["method"] == "kde":
        density_scorer = get_kernel_density_estimator(params["X"]).score_samples
        weight_matrix = get_weights_kde(params["n_samples"], params["X"], 
                                        params["distance_threshold"],
                                        params["distance_function"], 
                                        params["weight_function"],
                                        density_scorer)
    elif params["method"] == "epsilon":
        weight_matrix = get_weights_epsilon(params["n_samples"], params["X"], 
                                            params["distance_threshold"], 
                                            params["distance_function"], 
                                            params["weight_function"])

    elif params["method"] == "knn":
        weight_matrix = get_weights_knn(params["n_samples"], params["X"], 
                                        params["distance_threshold"], 
                                        params["n_neighbours"], 
                                        params["distance_function"], 
                                        params["weight_function"])

    return weight_matrix

def construct_graph(weight_matrix):
    graph = csr_matrix(weight_matrix)
    return graph

def find_shortest_path(graph, start_point_idx):
    """Get the shortest path from the start point to all other points.

    Args:
        graph (csr_matrix): a sparse matrix that contains the weight of the graph.
        start_point_idx (int): the index of the start point

    Returns:
        dist_matrix: the matrix of distances between graph nodes. It has shape (n_indices, n_nodes)
        predecessors: the matrix of predecessors to reconstruct the shortest paths. 
                It has shape (n_indices, n_nodes). Each entry predecessors[i, j] 
                gives the index of the previous node in the path from point i to point j.
    """

    # Compute the shortest distance path from start point to any other nodes.
    # dist_matrix has shape (n_indices, n_nodes).
    dist_matrix, predecessors = csgraph.dijkstra(
        csgraph=graph, directed=False, indices=start_point_idx, return_predecessors=True
    )

    # Need to avoid the shortest distance node to be the start point itself
    # dist_matrix[start_point_idx] = np.inf

    return dist_matrix, predecessors

def reconstruct_shortest_path(predecessors, start_point_idx, end_point_idx):
    """Get all the nodes along the path between the start point and the end point. 

    Args:
        predecessors (matrix of shape (1, n_nodes)): contain the previous node in the path.
        start_point_idx (int): the index of the start data point
        end_point_idx (int): the index of the end data point

    Returns:
        node_path (list): [start_point_idx, intermedium points index, end_point_idx]
    """

    if predecessors[end_point_idx] == start_point_idx:
        node_path = [end_point_idx]
    else:
        node_path = []
    intermedium_idx = end_point_idx
    while (predecessors[intermedium_idx] != start_point_idx):
        node_path.append(intermedium_idx)
        intermedium_idx = predecessors[intermedium_idx]
    node_path.append(start_point_idx)
    
    return node_path[::-1]
    

def reconstruct_shortest_path_all_candidates(predecessors, start_point_idx, 
                                             sorted_y_indices):
    # return the node paths to all valid CF points.
    # need to check that sorted_y_indices may be void.
    pass

def find_counterfactuals(start_point_value, X, y, graph,
                          target_class, predictor, density_scorer, params):

    start_point_idx = np.where((X == start_point_value).all(axis=1))[0][0]
    print(start_point_value)
    # positive predictions -> find indices of points in target class that predictor gives required confidence.
    y_proba = predictor.predict_proba(X)
    y_high_proba = np.where(y_proba >= params["prediction_threshold"])[0]
    y_target_indices = np.where(y.to_numpy() == target_class)[0]

    if params["method"] == "kde":
        density = np.exp(density_scorer(X))
        # Check the density requirement in Line 10, Algorithm 1.
        y_high_density = np.where(density >= params["density_threshold"])[0]
        y_candidate_indices = list(set(y_high_proba).intersection(set(y_target_indices)).intersection(set(y_high_density)))
    else:
        y_candidate_indices = list(set(y_high_proba).intersection(set(y_target_indices)))

    dist, predecessors = find_shortest_path(graph, start_point_idx)
    
    # change the distance matrix and only keep the distance to y_candidate_indices
    y_candidate_reachability = np.full(X.shape[0], 0)
    for idx in y_candidate_indices:
        y_candidate_reachability[idx] = 1.0
    
    # In the distance list, we keep only the distance to valid CF points.  
    y_candidate_dist = np.multiply(dist, y_candidate_reachability)
    # print(dist)
    # print(y_candidate_dist)
    # Sort the distance, and get the index of sorted valid CF points.
    valid_y_dist, = np.where(y_candidate_dist >0)
    # print("valid y dist: {}".format(valid_y_dist))
    sorted_y_indices = valid_y_dist[np.argsort(y_candidate_dist[valid_y_dist])]

    if len(sorted_y_indices) == 0:
        raise ValueError("No CF explanations available.")
    node_path = reconstruct_shortest_path(predecessors, start_point_idx, 
                                          sorted_y_indices[2])
    print(dist[node_path[-1]])
    print(node_path)
    return node_path, sorted_y_indices



def check_conditions(x_i, x_j):
    # check_conditions (function(numpy.ndarray, numpy.ndarray) -> Boolean): 
    #             check if additional custom conditions are satisfied
    return True


# TODO: add custom conditions
# TODO: sampling dataset to construct the graph.
# TODO: directed graph to simulate that you cannot reverse what you have done.


# Alternative 1:
# Use influence function as a proxy of density
# Use the number of outgoing lines (connectedness) of a node together with the local density.