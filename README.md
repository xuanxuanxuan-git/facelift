# FaceLift 

## Introduction
We introduce a <i>spatially-aware counterfactual explainers</i> in our proposed <i>explanatory multiverse</i> -- <b>FaceLift</b>. Our explainer encompasses all the possible counterfactual journeys, and show how to navigate, reason about and compare the geometry of these paths -- their affinity, branching, divergence and possible future convergence.


## Get started
``
pip install -r requirements.txt
``

## Run the explainer
- First, put the dataset file under the ``/data/raw_data/`` folder.
- Then run the following command with default configurations.

    ``
    python run_explainer.py
    ``

## MNIST dataset illustration in the paper
- The code to generate counterfactual explanations in the MNIST dataset can be viewed in [mnist_example.ipynb](/examples/mnist_example.ipynb). 


---

## Hyper-parameters
The hyper-parameters are defined in [params.yaml](/facelift/library/params.yaml) file. It plans to support inputs from command line. 

## Datasets
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