"""
Logic regarding sequential bootstrapping from chapter 4.
"""

from typing import List, Optional

import numpy as np
import pandas as pd
from numba import njit, prange


def get_ind_matrix(samples_info_sets: pd.Series, price_bars: pd.DataFrame) -> np.ndarray:
    """
    Build an Indicator Matrix

    Get indicator matrix. The book implementation uses bar_index as input, however there is no explanation
    how to form it. We decided that using triple_barrier_events and price bars by analogy with concurrency
    is the best option.

    Parameters
    ----------
    samples_info_sets : pd.Series
        Triple barrier events(t1) from `labeling.get_events()` method.
    price_bars : pd.DataFrame
        Price bars which were used to form triple barrier events.

    Returns
    -------
    np.ndarray
        Indicator binary matrix indicating what (price) bars influence the label for each observation

    Notes
    ---
    Reference: Advances in Financial Machine Learning, Snippet 4.3, page 65.
    """
    if bool(samples_info_sets.isnull().values.any()) is True or bool(samples_info_sets.index.isnull().any()) is True:
        raise ValueError("NaN values in `triple_barrier_events`. Drop NaN values to continue.")

    triple_barrier_events = pd.DataFrame(samples_info_sets)  # Convert Series to DataFrame

    # Take only period covered in triple_barrier_events
    trimmed_price_bars_index = price_bars[
        (price_bars.index >= triple_barrier_events.index.min()) & (price_bars.index <= triple_barrier_events.t1.max())
    ].index

    label_endtime = triple_barrier_events.t1
    bar_index = list(triple_barrier_events.index)  # Generate index for indicator matrix from t1 and index
    bar_index.extend(triple_barrier_events.t1)
    bar_index.extend(trimmed_price_bars_index)  # Add price bars index
    bar_index = sorted(list(set(bar_index)))  # Drop duplicates and sort

    # Get sorted timestamps with index in sorted array
    sorted_timestamps = dict(zip(sorted(bar_index), range(len(bar_index))))

    tokenized_endtimes = np.column_stack(
        (
            label_endtime.index.map(sorted_timestamps),
            label_endtime.map(sorted_timestamps).values,
        )
    )  # Create array of arrays: [label_index_position, label_endtime_position]

    ind_mat = np.zeros((len(bar_index), len(label_endtime)), dtype=np.int64)  # Init indicator matrix
    for sample_num, label_array in enumerate(tokenized_endtimes):
        label_index = label_array[0]
        label_endtime = label_array[1]
        ones_array = np.ones(
            (1, label_endtime - label_index + 1)
        )  # Ones array which corresponds to number of 1 to insert
        ind_mat[label_index : label_endtime + 1, sample_num] = ones_array
    return ind_mat


def get_ind_mat_average_uniqueness(ind_mat: np.ndarray) -> float:
    """
    Compute Average Uniqueness

    Average uniqueness from indicator matrix

    Parameters
    ----------
    ind_mat : np.ndarray
        Indicator binary matrix.

    Returns
    -------
    avg_uniqueness : float
        Average uniqueness.
    ---
    Reference: Advances in Financial Machine Learning, Snippet 4.4. page 65.
    """
    ind_mat = np.array(ind_mat, dtype=np.float64)
    concurrency = ind_mat.sum(axis=1)
    uniqueness = np.divide(ind_mat.T, concurrency, out=np.zeros_like(ind_mat.T), where=concurrency != 0)

    avg_uniqueness = uniqueness[uniqueness > 0].mean()

    return avg_uniqueness


def get_ind_mat_label_uniqueness(ind_mat: np.ndarray) -> np.ndarray:
    """
    Returns the indicator matrix element uniqueness.

    Parameters
    ----------
    ind_mat : np.ndarray
        Indicator binary matrix.

    Returns
    -------
    uniqueness : np.ndarray
        Element uniqueness.
    ---
    Reference: Advances in Financial Machine Learning, An adaption of Snippet 4.4. page 65.
    """
    ind_mat = np.array(ind_mat, dtype=np.float64)
    concurrency = ind_mat.sum(axis=1)
    uniqueness = np.divide(ind_mat.T, concurrency, out=np.zeros_like(ind_mat.T), where=concurrency != 0)
    return uniqueness


@njit(parallel=True)
def _bootstrap_loop_run(ind_mat: np.ndarray, prev_concurrency: np.ndarray) -> np.ndarray:
    """
    Part of Sequential Bootstrapping for-loop. Using previously accumulated concurrency array, loops through all samples
    and generates averages uniqueness array of label based on previously accumulated concurrency

    Parameters
    ----------
    ind_mat : np.ndarray
        Indicator matrix from get_ind_matrix function.
    prev_concurrency : np.ndarray
        Accumulated concurrency from previous iterations of sequential bootstrapping.

    Returns
    -------
    avg_unique : np.ndarray
        Label average uniqueness based on prev_concurrency.
    """
    avg_unique = np.zeros(ind_mat.shape[1], dtype=np.float64)  # Array of label uniqueness

    for i in prange(ind_mat.shape[1]):  # pylint: disable=not-an-iterable
        prev_average_uniqueness = 0
        number_of_elements = 0
        reduced_mat = ind_mat[:, i]
        for j in range(len(reduced_mat)):  # pylint: disable=consider-using-enumerate
            if reduced_mat[j] > 0:
                new_el = reduced_mat[j] / (reduced_mat[j] + prev_concurrency[j])
                average_uniqueness = (prev_average_uniqueness * number_of_elements + new_el) / (number_of_elements + 1)
                number_of_elements += 1
                prev_average_uniqueness = average_uniqueness
        avg_unique[i] = average_uniqueness
    return avg_unique


def seq_bootstrap(
    ind_mat: np.ndarray,
    sample_length: Optional[int] = None,
    warmup_samples: Optional[List[int]] = None,
    compare: bool = False,
    verbose: bool = False,
    random_state: np.random.RandomState = np.random.RandomState(),
) -> List[int]:
    """
    Return Sample from Sequential Bootstrap

    Generate a sample via sequential bootstrap.

    Parameters
    ----------
    ind_mat : np.ndarray
        Indicator matrix from triple barrier events.
    sample_length : Optional[int]
        Length of bootstrapped sample.
    warmup_samples : Optional[List[int]]
        List of previously drawn samples.
    compare : bool
        Flag to print standard bootstrap uniqueness vs sequential bootstrap uniqueness.
    verbose : bool
        Flag to print updated probabilities on each step.
    random_state : np.random.RandomState
        Random state

    Returns
    -------
    phi : List[int]
        Bootstrapped samples indexes

    Notes
    ---
    Moved from pd.DataFrame to np.matrix for performance increase.
    Reference: Advances in Financial Machine Learning, Snippet 4.5, Snippet 4.6, page 65.
    """

    if sample_length is None:
        sample_length = ind_mat.shape[1]

    if warmup_samples is None:
        warmup_samples = []

    phi = []  # Bootstrapped samples
    prev_concurrency = np.zeros(ind_mat.shape[0], dtype=np.float64)  # Init with zeros (phi is empty)
    while len(phi) < sample_length:
        avg_unique = _bootstrap_loop_run(ind_mat, prev_concurrency)
        prob = avg_unique / sum(avg_unique)  # Draw prob
        try:
            choice = warmup_samples.pop(0)  # It would get samples from warmup until it is empty
            # If it is empty from the beginning it would get samples based on prob from the first iteration
        except IndexError:
            choice = random_state.choice(range(ind_mat.shape[1]), p=prob)
        phi += [choice]
        prev_concurrency += ind_mat[:, choice]  # Add recorded label array from ind_mat
        if verbose is True:
            print(prob)

    if compare is True:
        standard_indx = np.random.choice(ind_mat.shape[1], size=sample_length)
        standard_unq = get_ind_mat_average_uniqueness(ind_mat[:, standard_indx])
        sequential_unq = get_ind_mat_average_uniqueness(ind_mat[:, phi])
        print("Standard uniqueness: {}\nSequential uniqueness: {}".format(standard_unq, sequential_unq))

    return phi
