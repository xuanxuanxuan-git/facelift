import warnings
warnings.filterwarnings(action="ignore", category=FutureWarning)
warnings.filterwarnings(action="ignore", category=UserWarning)

from carla.models.custom_model import Custom_MLP
from carla.recourse_methods import Face
from carla.recourse_methods.catalog.face.library.plotting import *
from carla.recourse_methods.catalog.face.configurator import Config
from generate_dataset import *
from utils.utils import *
from carla.models.negative_instances import predict_negative_instances
import argparse


if __name__ == '__main__':

    # parse the input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--explainer", "-e", type=str, default = "face", help="name of the CF explainer."
    )
    parser.add_argument(
        "--predictor", "-p", type=str, default="mlp", help="name of the predictor."
    )

    # Load arguments
    args, _ = parser.parse_known_args()
    config = Config(
        explainer=args.explainer, predictor=args.predictor).face_config_dict
    init_seed(config["seed"])

    df = create_synthetic_dataset() # load original dataset as a DataFrame
    dataset = load_dataset(df)  # standardisation
    model = Custom_MLP(dataset)

    # factuals = predict_negative_instances(model, dataset.df)
    
    target_factual_idx = find_instance(dataset.df)[0]
    target_factual = dataset.df.loc[[target_factual_idx]]

    cf_explainer = Face(model, config)

    counterfactuals = cf_explainer.get_counterfactuals(target_factual)
    # print(counterfactuals)
