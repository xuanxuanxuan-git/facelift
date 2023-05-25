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

## Hyper-parameters
The hyper-parameters are defined in [params.yaml](/facelift/library/params.yaml) file. It is currently not supporting inputs from command line. 

### MNIST dataset
<code>
start_point_idx=1<br>
target_class = 9<br>
distance_threshold: 6.1-7<br>
penalty_term: 1.1<br>
knn: {n_neighbours: 5/10/20}<br>
kde: n/a
</code>

### HELOC dataset
To use tabular dataset, we first need to do preprocessing
- One-hot encoding on categorical features
- Normalisation 


---

## Three methods proposed in FACE OG
- epsilon-graph: the most basic method, as long as neighbours are within epsilon-distance, they are kept in the adjacency matrix. The weight of the edge is diminished (logarithmically, by default) by the distance (default is l2-norm).

- knn-graph: for all neighbours which are within the distance-threshold, only top-k nearest neighbours are kept in the adjacency matrix. Therefore, this method runs additional selection process. 

- kde-graph: for all points that are within the distance-threshold, the value of the edge (dist) is weighted inversely by the kernel density.

## Known bugs in FACE OG

1. KNN method: after filtering out k neighbours, the weight matrix is asymmetric. But when FACE constructs graph, it will only keep the bottom left half of the matrix.
2. $\epsilon^d$ will always be greater than $x_i-x_j$. So the fraction will always be less than 1. This means if using inverse log, the edge weight will be negative!
3. Unsure: how to calculate kde in image datasets.