import warnings
warnings.filterwarnings(action="ignore", category=FutureWarning)
warnings.filterwarnings(action="ignore", category=UserWarning)

from carla.data.catalog import OnlineCatalog, CsvCatalog
# from carla import MLModelCatalog
from carla.models.custom_model import Custom_MLP
from carla.recourse_methods import Face
from carla.models.negative_instances import predict_negative_instances
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# positive class has label 1 (which should be red in Rafael's code)
def create_synthetic_dataset():
    # create synthetic dataset 
    t = 1
    t1 = 10 * np.random.random_sample(100 * t) - 0.50
    t0 = np.random.normal(0, 0.40, 100 * t)
    x1 = np.vstack((t0, t1)).T
    y1 = np.zeros(100 * t)

    t1 = 6 * np.random.random_sample(100 * t) - 0.50
    t0 = np.random.normal(0, 0.50, 100 * t)
    x2 = np.vstack((t1, t0)).T
    y2 = np.ones(100 * t)

    mean = [3.50, 8.00]
    sigma = 0.50
    cov = sigma * np.identity(2)
    n3 = 50 * t
    x3 = np.random.multivariate_normal(mean, cov, n3)
    y3 = np.ones(n3)

    X = np.concatenate((x1, x2, x3), axis=0)
    y = np.concatenate((y1, y2, y3))

    # fig, ax = plt.subplots()
    # ax.scatter(X[:,0], X[:,1], c=y)
    # plt.show()
    df = pd.DataFrame(X, columns = ['x1', 'x2'])
    df['y'] = y
  
    return df


def load_dataset(df):
    continuous = ['x1', 'x2']
    target = 'y'
    dataset = CsvCatalog(file_path=df, continuous=continuous, categorical=[], immutables=[], target=target)

    return dataset

def find_instance(X):
    target_df = X.loc[(X['x1']<=0.2) & (X['x2']>=0.8) & (X['x2']<0.9)]
    return target_df.index


if __name__ == '__main__':

    df = create_synthetic_dataset()
    dataset = load_dataset(df)
    model = Custom_MLP(dataset)

    # data_name = "adult"
    # dataset = OnlineCatalog(data_name)
    # model = MLModelCatalog(dataset, "ann", "pytorch") 

    factuals = predict_negative_instances(model, dataset.df)
    
    target_factual_idx = find_instance(factuals)[0]
    target_factual = factuals.loc[[target_factual_idx]]
    print(target_factual)

    hyperparams_face = {"mode": "epsilon", "fraction": 0.1}
    
    cf_explainer = Face(model, hyperparams_face)

    counterfactuals = cf_explainer.get_counterfactuals(target_factual)

    print(counterfactuals)
