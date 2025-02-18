"""
Implements the Combinatorial Purged Cross-Validation class from Chapter 12
"""

from itertools import combinations
from typing import List

import numpy as np
import pandas as pd
from scipy.special import comb
from sklearn.model_selection import KFold

from mlfinpy.cross_validation import ml_get_train_times


def _get_number_of_backtest_paths(n_train_splits: int, n_test_splits: int) -> float:
    """
    Number of combinatorial paths for CPCV(N,K)

    Parameters
    ----------
    n_train_splits : int
        Number of train splits.
    n_test_splits : int
        Number of test splits.

    Returns
    -------
    int
        Number of backtest paths for CPCV(N,k).
    """
    return int(comb(n_train_splits, n_train_splits - n_test_splits) * n_test_splits / n_train_splits)


class CombinatorialPurgedKFold(KFold):
    """
    Advances in Financial Machine Learning, Chapter 12.

    Implements Combinatorial Purged Cross Validation (CPCV).

    The train is purged of observations overlapping test-label intervals.
    Test set is assumed contiguous (shuffle=False), without training samples in between.

    Parameters
    ----------
    n_splits : int, optional
        The number of splits. Default is 3.
    samples_info_sets : pd.Series, optional
        The information range on which each record is constructed from.
        *samples_info_sets.index*: Time when the information extraction started.
        *samples_info_sets.value*: Time when the information extraction ended.
    pct_embargo : float, optional
        Percent that determines the embargo size. Default is 1.
    """

    def __init__(
        self, n_splits: int = 3, n_test_splits: int = 2, samples_info_sets: pd.Series = None, embargo: int = 1
    ):

        if not isinstance(samples_info_sets, pd.Series):
            raise ValueError("The samples_info_sets param must be a pd.Series")
        super(CombinatorialPurgedKFold, self).__init__(n_splits, shuffle=False, random_state=None)

        self.samples_info_sets = samples_info_sets
        self.embargo = embargo
        self.n_test_splits = n_test_splits
        self.num_backtest_paths = _get_number_of_backtest_paths(self.n_splits, self.n_test_splits)
        self.backtest_paths = []  # Array of backtest paths

    def _generate_combinatorial_test_ranges(self, splits_indices: dict) -> List:
        """
        Using start and end indices of test splits from KFolds and number of test_splits (self.n_test_splits),
        generates combinatorial test ranges splits.

        Parameters
        ----------
        splits_indices : dict
            Test fold integer index: [start test index, end test index].

        Returns
        -------
        list
            Combinatorial test splits ([start index, end index]).
        """

        # Possible test splits for each fold
        combinatorial_splits = list(combinations(list(splits_indices.keys()), self.n_test_splits))
        combinatorial_test_ranges = []  # List of test indices formed from combinatorial splits
        for combination in combinatorial_splits:
            temp_test_indices = []  # Array of test indices for current split combination
            for int_index in combination:
                temp_test_indices.append(splits_indices[int_index])
            combinatorial_test_ranges.append(temp_test_indices)
        return combinatorial_test_ranges

    def _fill_backtest_paths(self, train_indices: list, test_splits: list):
        """
        Using start and end indices of test splits and purged/embargoed train indices from CPCV, find backtest path and
        place in the path where these indices should be used.

        Parameters
        ----------
        train_indices : list
            List of train indices.
        test_splits : list
            List of lists with first element corresponding to test start index and second - test end.
        """
        # Fill backtest paths using train/test splits from CPCV
        for split in test_splits:
            found = False  # Flag indicating that split was found and filled in one of backtest paths
            for path in self.backtest_paths:
                for path_el in path:
                    if path_el["train"] is None and split == path_el["test"] and found is False:
                        path_el["train"] = np.array(train_indices)
                        path_el["test"] = list(range(split[0], split[-1]))
                        found = True

    # noinspection PyPep8Naming
    def split(self, X: pd.DataFrame, y: pd.Series = None, groups=None):
        """
        The main method to call for the PurgedKFold class.

        Parameters
        ----------
        X : pd.DataFrame
            Samples dataset that is to be split.
        y : pd.Series, optional
            Sample labels series.
        groups : array-like, optional
            Group labels for the samples used while splitting the dataset into train/test set.

        Returns
        -------
        tuple
            [train list of sample indices, and test list of sample indices].
        """
        if X.shape[0] != self.samples_info_sets.shape[0]:
            raise ValueError("X and the 'samples_info_sets' series param must be the same length")

        test_ranges = [(ix[0], ix[-1] + 1) for ix in np.array_split(np.arange(X.shape[0]), self.n_splits)]
        splits_indices = {}
        for index, [start_ix, end_ix] in enumerate(test_ranges):
            splits_indices[index] = [start_ix, end_ix]

        combinatorial_test_ranges = self._generate_combinatorial_test_ranges(splits_indices)
        # Prepare backtest paths
        for _ in range(self.num_backtest_paths):
            path = []
            for split_idx in splits_indices.values():
                path.append({"train": None, "test": split_idx})
            self.backtest_paths.append(path)

        for test_splits in combinatorial_test_ranges:

            # Embargo
            self.embargo = 0
            delta = self.samples_info_sets[0] - self.samples_info_sets.index[0]
            embargo = delta * self.embargo

            test_times = pd.Series(
                index=[self.samples_info_sets.index[ix[0]] for ix in test_splits],
                data=[
                    (
                        self.samples_info_sets[ix[1] - 1]
                        if ix[1] + 1 >= X.shape[0]
                        else self.samples_info_sets[ix[1] - 1] + embargo
                    )
                    for ix in test_splits
                ],
            )
            test_indices = []
            for [start_ix, end_ix] in test_splits:
                test_indices.extend(list(range(start_ix, end_ix)))

            # Purge
            train_times = ml_get_train_times(self.samples_info_sets, test_times)

            # Get indices
            train_indices = []
            # for train_ix in train_times.index:
            # train_indices.append(self.samples_info_sets.index.get_loc(train_ix))
            for train_ix in train_times.index.unique():
                loc = self.samples_info_sets.index.get_loc(train_ix)
                if not isinstance(loc, int):
                    loc = np.arange(loc.start, loc.stop)
                train_indices.append(loc)

            self._fill_backtest_paths(train_indices, test_splits)

            train_indices = np.concatenate(train_indices)
            test_indices = np.array(test_indices)

            if len(np.intersect1d(train_indices, test_indices)) > 0:
                raise Exception("Train and test intersect")

            yield train_indices, test_indices

        all_train = []
        all_test = []
        np.concatenate((train_indices, test_indices))
        all_train.append(train_indices)
        all_test.append(test_indices)

        all_train = np.concatenate((all_train, train_indices))
        all_test = np.concatenate((all_test, test_indices))

        t = np.arange(0, len(X))
        np.isin(t, all_train, invert=True).any()

        t[np.isin(t, all_train, invert=True)]
        all_train[np.isin(all_train, t, invert=True)]
