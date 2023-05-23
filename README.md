# FaceLift 
## Spatially-aware counterfactual explainers

---
## Get started
``
pip install -r requirements.txt
``

## Run the code
``
python run_explainer.py
``


## Three methods proposed in FACE
- epsilon-graph: the most basic method, as long as neighbours are within epsilon-distance, they are kept in the adjacency matrix. The weight of the edge is diminished (logarithmically, by default) by the distance (default is l2-norm).

- knn-graph: for all neighbours which are within the distance-threshold, only top-k nearest neighbours are kept in the adjacency matrix. Therefore, this method runs additional selection process. 

- kde-graph: for all points that are within the distance-threshold, the value of the edge (dist) is weighted inversely by the kernel density.

## Known bugs in FACE

1. KNN method: after filtering out k neighbours, the weight matrix is asymmetric. But when FACE constructs graph, it will only keep the bottom left half of the matrix.
2. $\epsilon^d$ will always be greater than $x_i-x_j$. So the fraction will always be less than 1. This means if using inverse log, the edge weight will be negative!
3. Unsure: how to calculate kde in image datasets.