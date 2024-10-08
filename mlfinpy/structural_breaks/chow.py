"""
Explosiveness tests: Chow-Type Dickey-Fuller Test
"""

import pandas as pd

from mlfinpy.structural_breaks.sadf import get_betas
from mlfinpy.util import mp_pandas_obj

# pylint: disable=invalid-name


def _get_dfc_for_t(series, molecule):
    """
    Get Chow-Type Dickey-Fuller Test statistics for each index in molecule

    Parameters
    ----------
    series : pd.Series
        Series to test.
    molecule : list
        Dates to test.

    Returns
    -------
    pd.Series
        Statistics for each index from molecule.
    """

    dfc_series = pd.Series(index=molecule, dtype="float64")

    for index in molecule:
        series_diff = series.diff().dropna()
        series_lag = series.shift(1).dropna()
        series_lag[:index] = 0  # D_t* indicator: before t* D_t* = 0

        y = series_diff.loc[series_lag.index].values
        x = series_lag.values
        coefs, coef_vars = get_betas(x.reshape(-1, 1), y)
        b_estimate, b_var = coefs[0], coef_vars[0][0]
        dfc_series[index] = b_estimate / (b_var**0.5)

    return dfc_series


def get_chow_type_stat(
    series: pd.Series, min_length: int = 20, num_threads: int = 8, verbose: bool = True
) -> pd.Series:
    """
    Multithread implementation of Chow-Type Dickey-Fuller Test, p.251-252

    Parameters
    ----------
    series : pd.Series
        Series to test.
    min_length : int, optional
        Minimum sample length used to estimate statistics.
    num_threads : int, optional
        Number of cores to use.
    verbose : bool, optional
        Flag to report progress on asynch jobs.

    Returns
    -------
    pd.Series
        Chow-Type Dickey-Fuller Test statistics.
    """
    # Indices to test. We drop min_length first and last values
    molecule = series.index[min_length : series.shape[0] - min_length]
    dfc_series = mp_pandas_obj(
        func=_get_dfc_for_t,
        pd_obj=("molecule", molecule),
        series=series,
        num_threads=num_threads,
        verbose=verbose,
    )
    return dfc_series
