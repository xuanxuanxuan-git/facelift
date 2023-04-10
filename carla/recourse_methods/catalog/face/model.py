from typing import Dict, Optional

import pandas as pd
import numpy as np
from carla.models.api import MLModel
from carla.recourse_methods.api import RecourseMethod
# from carla.recourse_methods.catalog.face.library import graph_search
from carla.recourse_methods.catalog.face.library.face_helpers import *
from carla.recourse_methods.catalog.face.library.plotting import *
from carla.recourse_methods.processing import (
    check_counterfactuals,
    # encode_feature_names,
    merge_default_parameters,
)


class Face(RecourseMethod):
    """
    Implementation of FACE from Poyiadzi et.al. [1].

    Parameters
    ----------
    mlmodel : carla.model.MLModel
        Black-Box-Model
    hyperparams : dict
        Dictionary containing hyperparameters. See notes below for its contents.

    Methods
    -------
    get_counterfactuals:
        Generate counterfactual examples for given factuals.

    Notes
    -----
    - Hyperparams
        Hyperparameter contains important information for the recourse method to initialize.
        Please make sure to pass all values as dict with the following keys.

        * "method": {"kde", "knn", "epsilon"},
            Decides over type of FACE
        * "num_of_paths": int [>0]
            number of paths required as final counterfactual solutions
        * "weight_function": function
            the function to calculate the weight of an edge in a graph
        * "prediction_threshold": float [0 =< x =< 1]
            a lower bound on the prediction confidence from the predictor.
            (t_p in line 10, Algorithm 1)
        * "density_threshold": float [0 < x < 1]
            density threshold t_d used in line 10, Algorithm 1. 
            This hyper-parameter is used only in "kde" method.
        * "distance_threshold": float
            distance threshold used in line 2, Algorithm 1
        * "radius_limit": 
            eta_d, the volume of a sphere with a unit radius in R^d
        * "n_neighbours": int [0< x < n_samples]
            k in equation (3) for k-NN graph
        * "distance_function": function
            distance function d; currently support l1 and l2; default is l2
    """

    _DEFAULT_NUM_OF_PATHS = 5
    _DEFAULT_METHOD = 'kde'
    _DEFAULT_WEIGHT_FUNCTION = lambda x: -np.log(x)
    _DEFAULT_PREDICTION_THRESHOLD = 0.6
    _DEFAULT_DENSITY_THRESHOLD = 1e-5
    _DEFAULT_DISTANCE_THRESHOLD = 1.1
    _DEFAULT_RADIUS_LIMIT = 1.10
    _DEFAULT_N_NEIGHBOURS = 5   #10
    _DEFAULT_DISTANCE_FUNCTION = "l2"

    _DEFAULT_HYPERPARAMS = {"method": _DEFAULT_METHOD, 
                            "num_of_paths": _DEFAULT_NUM_OF_PATHS,
                            "prediction_threshold": _DEFAULT_PREDICTION_THRESHOLD,
                            "weight_function": _DEFAULT_WEIGHT_FUNCTION,
                            "density_threshold": _DEFAULT_DENSITY_THRESHOLD,
                            "distance_threshold": _DEFAULT_DISTANCE_THRESHOLD,
                            "radius_limit": _DEFAULT_RADIUS_LIMIT,
                            "n_neighbours": _DEFAULT_N_NEIGHBOURS,
                            "distance_function": _DEFAULT_DISTANCE_FUNCTION}

    def __init__(self, mlmodel: MLModel, hyperparams: Optional[Dict] = None) -> None:
        super().__init__(mlmodel)

        # The output checked_hyperparams ensures that all the params are set, 
        # i.e., none of the hyperparameters are void.
        checked_hyperparams = merge_default_parameters(
            hyperparams, self._DEFAULT_HYPERPARAMS
        )
        self.method = checked_hyperparams["method"]
        self.num_of_paths = checked_hyperparams["num_of_paths"]
        self.distance_threshold = checked_hyperparams["distance_threshold"]
        self.prediction_threshold = checked_hyperparams["prediction_threshold"]
        self.weight_function = checked_hyperparams["weight_function"]
        self.density_threshold = checked_hyperparams["density_threshold"]
        self.radius_limit = checked_hyperparams["radius_limit"]
        self.n_neighbours = checked_hyperparams["n_neighbours"]
        self.distance_function = checked_hyperparams["distance_function"]

        # self._immutables = encode_feature_names(
        #     self._mlmodel.data.immutables, self._mlmodel.feature_input_order
        # )

    ## --------------------------------------------------------------- ##
    # Check that the value of each parameter is valid.
    @property
    def method(self) -> str:
        """
        return the method to build the graph
        return: str
        """
        return self._method

    @method.setter
    def method(self, method: str) -> None:
        if method in ["knn", "epsilon", "kde"]:
            self._method = method
        else:
            raise ValueError("Mode has to be either knn, epsilon, or kde.")

    @property
    def num_of_paths(self) -> int:
        """
        return the number of paths needed
        return: int
        """
        return self._num_of_paths
    
    @num_of_paths.setter
    def num_of_paths(self, num_of_paths: int) -> None:
        # Check that each instance has to be a positive integer
        if isinstance(num_of_paths, int) and num_of_paths > 0:
            self._num_of_paths = num_of_paths
        else:
            raise ValueError("Number of paths has to be a positive integer")

    @property
    def weight_function(self):
        """
        return the weight function to calculate the weight of a edge in the graph
        """
        return self._weight_function

    @weight_function.setter
    def weight_function(self, weight_function):
        self._weight_function = weight_function
    
    @property
    def prediction_threshold(self) -> float:
        """
        return the lower bound on the prediction confidence from the predictor
        """
        return self._prediction_threshold

    @prediction_threshold.setter
    def prediction_threshold(self, prediction_threshold) -> None:
        if prediction_threshold >= 0 and prediction_threshold <= 1:
            self._prediction_threshold = prediction_threshold
        else:
            raise ValueError("Prediction threshold should be in range [0,1] inclusive.")
    
    @property
    def density_threshold(self) -> float:
        """
        return density threshold t_d. used in "kde" method only.
        """
        return self._density_threshold

    @density_threshold.setter
    def density_threshold(self, density_threshold) -> None:
        assert density_threshold >= 0 and density_threshold <= 1, \
                "Density threshold must be in range [0,1]"
        self._density_threshold = density_threshold
        
    @property
    def distance_threshold(self) -> float:
        """
        distance threshold used in line 2, Algorithm 1
        """
        return self._distance_threshold
    
    @distance_threshold.setter
    def distance_threshold(self, distance_threshold) -> None:
        #TODO: what's the value range of epsilon? 
        # assert distance_threshold >= 0 and distance_threshold <= 1,\
        #        "Distance threshold should be in range [0,1] inclusive"
        self._distance_threshold = distance_threshold
    
    @property
    def radius_limit(self) -> float:
        """
        return etd_d, the volume of a sphere with a unit radius in R^d
        """
        return self._radius_limit

    @radius_limit.setter
    def radius_limit(self, radius_limit) -> None:
        assert radius_limit > 0, \
                "radius limit should be greater than 0."
        self._radius_limit = radius_limit

    @property
    def n_neighbours(self) -> int:
        """
        return parameter num_neighbours in knn graph
        """
        return self._n_neighbours

    @n_neighbours.setter
    def n_neighbours(self, n_neighbours) -> None:
        assert isinstance(n_neighbours, int) and n_neighbours > 0, \
            "num_neighbours should be a positive integer"
        self._n_neighbours = n_neighbours

    @property
    def distance_function(self):
        """
        return the distance function d
        """
        return self._distance_function 
    
    @distance_function.setter
    def distance_function(self, distance_function):

        if distance_function == "l1":
            self._distance_function = lambda x_0, x_1: np.linalg.norm(x_0-x_1, ord=1)
        elif distance_function == "l2": 
            self._distance_function = lambda x_0, x_1: np.linalg.norm(x_0-x_1)
        else:
            raise ValueError("Distance function is not supported. Available distance functions are \"l1\" and \"l2\".")
    
    
    def get_counterfactuals(self, factuals: pd.DataFrame) -> pd.DataFrame:
       
        df = self._mlmodel.data.df.copy()
        self.X = df.iloc[:, :-1]
        self.y = df.iloc[:, -1]

        self.n_samples, self.n_features = self.X.shape
        
        self.method = 'knn'
        self.distance_threshold = 0.6
        self.n_neighbours = 6          # only used in "knn" method. 
        self.prediction_threshold = 0.6 # only used after constructing the graph
        self.density_threshold = 1e-5   # only used in "kde" after constructing the graph
        
        params = {
            "X": self.X,
            "method": self.method,
            "n_samples": self.n_samples,
            "distance_threshold": self.distance_threshold,
            "distance_function": self.distance_function,
            "weight_function": self.weight_function,
            "n_neighbours": self.n_neighbours   # used only in "knn" method
        }
    
        # Construct the graph    
        weight_matrix = get_weight_matrix(params)

        graph = construct_graph(weight_matrix)

        # define more hyper-parameters    
        start_point_value = self.X.iloc[20]
        target_class = 1
        params["prediction_threshold"] = self.prediction_threshold
        if params["method"] == "kde":
            params["density_threshold"] = self.density_threshold
        
        density_scorer = get_kernel_density_estimator(self.X).score_samples
        
        node_path, sorted_cf_indices = find_counterfactuals(start_point_value, self.X, 
                                         self.y, graph, target_class, 
                                         self._mlmodel, density_scorer, params)

        plot_counterfactual_explanations(df, self.X, self._mlmodel, 1, node_path, sorted_cf_indices)

        exit()



        # >drop< factuals from dataset to prevent duplicates,
        # >reorder< and >add< factuals to top; necessary in order to use the index
        cond = df.isin(factuals).values
        df = df.drop(df[cond].index)
        df = pd.concat([factuals, df], ignore_index=True)

        df = self._mlmodel.get_ordered_features(df)
        factuals = self._mlmodel.get_ordered_features(factuals)

        list_cfs = []
        for i in range(factuals.shape[0]):
            cf = graph_search(
                df,
                i,
                self._immutables,
                self._mlmodel,
                mode=self._method,
                frac=self._fraction,
            )
            list_cfs.append(cf)
        print("list of counterfactuals are: {}".format(list_cfs))
        df_cfs = check_counterfactuals(self._mlmodel, list_cfs, factuals.index)
        df_cfs = self._mlmodel.get_ordered_features(df_cfs)
        return df_cfs
