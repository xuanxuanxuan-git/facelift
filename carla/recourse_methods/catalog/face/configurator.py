import os
import yaml
import re

# TODO: parse the arguments from the input.

class Config(object):

    # Load internal configuration file
    def __init__(self, explainer=None, predictor=None):
        self.yaml_loader = self._build_yaml_loader()
        self.face_config_dict = self._load_internal_config_file()

    @property
    def face_config_dict(self) -> dict:
        return self._face_config_dict
    
    @face_config_dict.setter
    def face_config_dict(self, face_config_dict) -> None:
        self._face_config_dict = face_config_dict
    
    def _build_yaml_loader(self):
        loader = yaml.FullLoader
        loader.add_implicit_resolver(
            "tag:yaml.org,2002:float",
            re.compile(
                """^(?:
                [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
            |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
            |\\.[0-9_]+(?:[eE][-+][0-9]+)?
            |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
            |[-+]?\\.(?:inf|Inf|INF)
            |\\.(?:nan|NaN|NAN))$""",
                re.X,
            ),
            list("-+0123456789."),
        )
        return loader

    def _load_internal_config_file(self):
        """Load the params.yaml file

        Returns:
            dict(): hyperparameter configuration dictionary
        """
        current_path = os.path.dirname(os.path.realpath(__file__))
        face_params_path = os.path.join(current_path, "library/params.yaml")
        face_config_dict = dict()
        with open(face_params_path, "r", encoding="utf-8") as f:
            face_config_dict.update(
                yaml.load(f.read(), Loader=self.yaml_loader)
            )
        return face_config_dict

