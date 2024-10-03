"""
This module contains an implementation of an exponentially weighted moving average based on sample size.
The inspiration and context for this code was from a blog post by writen by Maksim Ivanov:
https://towardsdatascience.com/financial-machine-learning-part-0-bars-745897d4e4ba
"""

# Imports
import numpy as np
from numba import njit


@njit(nogil=True)
def ewma(arr_in: np.ndarray, window: int) -> np.ndarray:
    """
    Exponentially weighted moving average specified by a decay `window` to provide better adjustments
    for small windows via:

    .. math::
        y[t] = \\frac{x[t] + (1-a)x[t-1] + (1-a)^2x[t-2] + \\ldots + (1-a)^nx[t-n]}{1 + (1-a) + (1-a)^2 +
        \\ldots + (1-a)^n}

    Parameters
    ----------
    arr_in : np.ndarray
        A single dimensional numpy array.
    window : int
        The decay window, or 'span'.

    Returns
    -------
    ewma_arr : np.ndarray
        The EWMA vector, same length / shape as `arr_in`
    """

    arr_length: int = arr_in.shape[0]
    ewma_arr: np.ndarray = np.empty(arr_length, dtype=np.float64)
    alpha: float = 2 / (window + 1)
    weight: float = 1
    ewma_old: float = arr_in[0]
    ewma_arr[0] = ewma_old
    for i in range(1, arr_length):
        weight += (1 - alpha) ** i
        ewma_old = ewma_old * (1 - alpha) + arr_in[i]
        ewma_arr[i] = ewma_old / weight

    return ewma_arr
