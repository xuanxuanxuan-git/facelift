# My implementation of FACE

## TODO
- Create a function to connect hops in the path and plot multiple paths in one graph.

## Get started
- follow the installation in [CARLA](https://github.com/carla-recourse/CARLA).
  
<code>python run_explainers.py</code>

## Main files
- FACE model is implement in `carla/recourse_methods/catalog/face`.
- `library/params.yaml` defines all the hyper-parameters.
- `library/plotting.py` contains all code to for plotting (Documentation missing).
- `library/face_helpers.py` contains all critical code.
- `configurator.py` parses the params.yaml file and initialise hyper-parameters.
- `model.py` initialises the FACE model.

## Known bugs
- CF paths will not go right even after I relax all constraints.

<br>
<br>
---

## Notes for debugging, please ignore

### KNN method set-up

This will go right
- distance_threshold: 0.6
- n_neighbours: 6
- prediction_threshold: 0.6

This will go down
- distance_threshold: 0.6
- n_neighbours: 5
- prediction_threshold: 0.6

This will fail to give a solution
- distance_threshold: 0.6
- n_neighbours: 1
- prediction_threshold: 0.6

---
### Epsilon method set-up

This will go right
- distance_threshold: 0.2
- prediction_threshold: 0.6
  
This will go down
- distance_threshold: 0.1
- prediction_threshold: 0.6

This will fail
- distance_threshold: 
- prediction_threshold: 0.6
