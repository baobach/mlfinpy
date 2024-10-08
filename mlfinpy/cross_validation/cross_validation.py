"""
Implements Сhapter 7 of AFML on Cross Validation for financial data.

Also Stacked Purged K-Fold cross validation and Stacked ml cross val score. These functions are used
for multi-asset datasets.
"""

from typing import Callable

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.metrics import log_loss
from sklearn.model_selection import BaseCrossValidator, KFold
from sklearn.utils import indexable
from sklearn.utils.validation import _num_samples


def ml_get_train_times(samples_info_sets: pd.Series, test_times: pd.Series) -> pd.Series:
    # pylint: disable=invalid-name
    """
    Purging observations in the training set

    This function find the training set indexes given the information on which each record is based
    and the range for the test set.
    Given test_times, find the times of the training observations.

    Parameters
    ----------
    samples_info_sets : pd.Series
        The information range on which each record is constructed from
        * ``samples_info_sets.index``: Time when the information extraction started.
        * ``samples_info_sets.value``: Time when the information extraction ended.
    test_times : pd.Series
        Times for the test dataset.

    Returns
    -------
    pd.Series
        Training set.
    """
    train = samples_info_sets.copy(deep=True)
    for start_ix, end_ix in test_times.items():
        df0 = train[(start_ix <= train.index) & (train.index <= end_ix)].index.unique()  # Train starts within test
        df1 = train[(start_ix <= train) & (train <= end_ix)].index.unique()  # Train ends within test
        df2 = train[(train.index <= start_ix) & (end_ix <= train)].index.unique()  # Train envelops test
        train = train.drop(df0.union(df1).union(df2))
    return train


class PurgedKFold(KFold):
    """
    Extend ``KFold`` class to work with labels that span intervals.

    The train is purged of observations overlapping test-label intervals.
    Test set is assumed contiguous (``shuffle`` = False), w/o training samples in between.

    Parameters
    ----------
    n_splits : int, default 3
        The number of splits.
    samples_info_sets : pd.Series, default None
        The information range on which each record is constructed from
        * ``samples_info_sets.index``: Time when the information extraction started.
        * ``samples_info_sets.value``: Time when the information extraction ended.
    pct_embargo : float
        Percent that determines the embargo size.
    """

    def __init__(self, n_splits: int = 3, samples_info_sets: pd.Series = None, pct_embargo: float = 0.0):

        if not isinstance(samples_info_sets, pd.Series):
            raise ValueError("The samples_info_sets param must be a ``pd.Series``")
        super(PurgedKFold, self).__init__(n_splits, shuffle=False, random_state=None)

        self.samples_info_sets = samples_info_sets
        self.pct_embargo = pct_embargo

    # noinspection PyPep8Naming
    def split(self, X: pd.DataFrame, y: pd.Series = None, groups: np.ndarray = None):
        """
        The main method to call for the PurgedKFold class

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        y : array-like of shape (n_samples,), default=None
            The target variable for supervised learning problems.

        groups : array-like of shape (n_samples,), default=None
            Group labels for the samples used while splitting the dataset into
            train/test set.

        Yields
        ------
        train : ndarray
            The training set indices for that split.

        test : ndarray
            The testing set indices for that split.
        """
        X, y, groups = indexable(X, y, groups)

        if X.shape[0] != self.samples_info_sets.shape[0]:
            raise ValueError("X and the 'samples_info_sets' series param must be the same length")

        indices = np.arange(_num_samples(X))

        embargo = int(X.shape[0] * self.pct_embargo)

        test_ranges = [(ix[0], ix[-1] + 1) for ix in np.array_split(np.arange(X.shape[0]), self.n_splits)]

        for start_ix, end_ix in test_ranges:
            test_indices = indices[start_ix:end_ix]

            if end_ix < X.shape[0]:
                end_ix += embargo

            test_times = pd.Series(
                index=[self.samples_info_sets.iloc[start_ix]], data=[self.samples_info_sets.iloc[end_ix - 1]]
            )
            train_times = ml_get_train_times(self.samples_info_sets, test_times)

            train_indices = list()

            for train_ix in train_times.index:
                train_indices.append(self.samples_info_sets.index.get_loc(train_ix))

            # Sanity check (no overlap)
            if len(np.intersect1d(train_indices, test_indices)) > 0:
                raise Exception("Train and test intersect")

            yield train_indices, test_indices


# noinspection PyPep8Naming
def ml_cross_val_score(
    classifier: ClassifierMixin,
    X: pd.DataFrame,
    y: pd.Series,
    cv_gen: BaseCrossValidator,
    sample_weight_train: np.ndarray = None,
    sample_weight_score: np.ndarray = None,
    scoring: Callable[[np.array, np.array], float] = log_loss,
) -> np.ndarray:
    """
    Using the PurgedKFold Class

    Function to run a cross-validation evaluation of the using sample weights and a custom CV generator.

    Note: This function is different to the book in that it requires the user to pass through a CV object. The book
    will accept a None value as a default and then resort to using PurgedCV, this also meant that extra arguments had to
    be passed to the function. To correct this we have removed the default and require the user to pass a CV object to
    the function.

    Parameters
    ----------
    classifier : ClassifierMixin
        A sk-learn Classifier object instance.
    X : pd.DataFrame
        The feature matrix of records to evaluate.
    y : pd.Series
        Ther target vector corresponding to the X dataset.
    cv_gen : BaseCrossValidator
        Cross Validation generator object instance.
    sample_weight_train : np.array
        Sample weights used to train the model for each record in the dataset.
    sample_weight_score : np.array
        Sample weights used to evaluate the model quality.
    scoring : Callable
        A metric scoring, can be custom sklearn metric.

    Returns
    -------
    np.array
        The computed score.

    Example
    -------
    >>> cv_gen = PurgedKFold(n_splits=3, samples_info_sets=samples_info_sets, pct_embargo=.1)
    >>> ml_cross_val_score(classifier, X, y, cv_gen, sample_weight_train=sample_train,
    >>>                    sample_weight_score=sample_score, scoring=accuracy_score)

    Notes
    -----
    Advances in Financial Machine Learning, Snippet 7.1, page 106.
    """

    # If no sample_weight then broadcast a value of 1 to all samples (full weight).
    if sample_weight_train is None:
        sample_weight_train = np.ones((X.shape[0],))

    if sample_weight_score is None:
        sample_weight_score = np.ones((X.shape[0],))

    # Score model on KFolds
    ret_scores = []
    for train, test in cv_gen.split(X=X, y=y):
        fit = classifier.fit(X=X.iloc[train, :], y=y.iloc[train], sample_weight=sample_weight_train[train])
        if scoring == log_loss:
            prob = fit.predict_proba(X.iloc[test, :])
            score = -1 * scoring(
                y.iloc[test], prob, sample_weight=sample_weight_score[test], labels=classifier.classes_
            )
        else:
            pred = fit.predict(X.iloc[test, :])
            score = scoring(y.iloc[test], pred, sample_weight=sample_weight_score[test])
        ret_scores.append(score)
    return np.array(ret_scores)
