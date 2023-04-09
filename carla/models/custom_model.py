from carla import MLModel
from sklearn.neural_network import MLPClassifier


class Custom_MLP(MLModel):
    def __init__(self, data):
        super().__init__(data)

        df_train = self.data.df_train
        df_test = self.data.df_test

        x_train = df_train[self.data.continuous]
        y_train = df_train[self.data.target]
        x_test = df_test[self.data.continuous]
        y_test = df_test[self.data.target]

        self._feature_input_order = self.data.continuous

        params = {
            "hidden_layer_sizes": (10,),
            "solver": "lbfgs",
            "alpha": 1e-3,
            "random_state": 41
        }

        self._mymodel = MLPClassifier(**params)
        self._mymodel.fit(
            x_train, y_train
        )
    @property
    def feature_input_order(self):
        return self._feature_input_order
    
    @property
    def backend(self):
        return "pytorch"

    @property
    def raw_model(self):
        return self._mymodel    
    
    def predict(self, x):
        return self._mymodel.predict(self.get_ordered_features(x))
    
    def predict_proba(self, x):
        return self._mymodel.predict_proba(self.get_ordered_features(x))