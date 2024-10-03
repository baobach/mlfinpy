"""
This module generates  synthetic classification dataset of INFORMED, REDUNDANT, and NOISE explanatory
variables based on the book Machine Learning for Asset Manager (code snippet 6.1)
"""

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification


# pylint: disable=invalid-name
def get_classification_data(
    n_features: int = 100,
    n_informative: int = 25,
    n_redundant: int = 25,
    n_samples: int = 10000,
    random_state: int = 0,
    sigma: float = 0.0,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    A function to generate synthetic classification datasets.

    This function is based on the book Machine Learning for Asset Manager (code snippet 6.1)
    to generate synthetic data. The output data is the feature matrix `X` and the label `y`.

    Parameters
    ----------
    n_features : int
        Total number of features to be generated (i.e. informative + redundant + noisy).
    n_informative : int
        Number of informative features.
    n_redundant : int
        Number of redundant features.
    n_samples : int
        Number of samples (rows) to be generate.
    random_state : int
        Random seed.
    sigma : float
        This argument is used to introduce substitution effect to the redundant features in
        the dataset by adding gaussian noise. The lower the  value of  sigma, the  greater the
        substitution effect.

    Returns
    -------
    X : pd.DataFrame
        Features to be used for training a machine learning model.
    y : pd.Series
        Labels corresponding to each sample.
    """
    np.random.seed(random_state)
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features - n_redundant,
        n_informative=n_informative,
        n_redundant=0,
        shuffle=False,
        random_state=random_state,
    )
    cols = ["I_" + str(i) for i in range(n_informative)]
    cols += ["N_" + str(i) for i in range(n_features - n_informative - n_redundant)]
    X, y = pd.DataFrame(X, columns=cols), pd.Series(y)
    i = np.random.choice(range(n_informative), size=n_redundant)
    for k, j in enumerate(i):
        X["R_" + str(k)] = X["I_" + str(j)] + np.random.normal(size=X.shape[0]) * sigma
    return X, y
