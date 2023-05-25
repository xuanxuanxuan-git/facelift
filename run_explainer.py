import argparse
from utils.utils import *
from utils.configurator import *
from data.preprocess import *
from predictors.predictors import mlp_classifier
from facelift.facelift_model import FaceLift

if __name__ == '__main__':

    # Parse the input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictor", "-p", type=str, default="mlp", help="name of the predictor."
    )
    parser.add_argument(
        "--dataset", "-d", type=str, default="heloc", help="dataset name."
    )

    # Load arguments
    args, _ = parser.parse_known_args()
    # Add input arguments to the configuration file
    config = Config(predictor=args.predictor).face_config_dict
    init_seed(config["seed"])

    # Load the processed dataset
    X, y = load_dataset("heloc")
    print(X)

    # Load predictor
    model = mlp_classifier(X, y).fitted_model
    predictions = model.predict_proba(X)

    # Define target point (input)
    start_point_idx = 1
    target_class = 9
    num_of_paths = 20

    # Initialise the facelift explainer
    cf_explainer = FaceLift(X, config)

    # Print counterfactual explanations
    cf_paths = cf_explainer.get_counterfactuals(X, y, model, predictions, start_point_idx, target_class, num_of_paths)

