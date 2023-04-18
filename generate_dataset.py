import numpy as np
import pandas as pd
from carla.data.catalog import CsvCatalog


def create_synthetic_dataset():

    # create synthetic dataset (copied from the OG code for reproducibility).
    # positive class has label 1 (which should be red in Rafael's code)

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

    df = pd.DataFrame(X, columns = ['x1', 'x2'])
    df['y'] = y
  
    return df


def load_dataset(df):
    """Load the dataset in CARLA framework. Define continuous/categorical/immutable data.

    Args:
        df (DataFrame): dataset

    Returns:
        class: CsvCatalog class
    """
    continuous = ['x1', 'x2']
    categorical = []
    immutable = []
    target = 'y'
    dataset = CsvCatalog(file_path=df, continuous=continuous, 
                         categorical=categorical, immutables=immutable, 
                         target=target)

    return dataset

def find_instance(X):
    """Manually find the factual point which needs CF explanations.

    Args:
        X (DataFrame): dataset

    Returns:
        int: index of the factual point
    """
    target_df = X.loc[(X['x1']>=0.453) & (X['x1']<=0.55) & (X['x2']>=0.8) & (X['x2']<0.85)]
    return target_df.index
