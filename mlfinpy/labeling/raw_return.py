"""
Raw Returns Labeling Method

Most basic form of labeling based on raw return of each observation relative to its previous value.
"""

from typing import Optional, Union

import numpy as np
import pandas as pd


def raw_return(
    prices: Union[pd.Series, pd.DataFrame],
    binary: bool = False,
    logarithmic: bool = False,
    resample_by: Optional[str] = None,
    lag: bool = True,
) -> Union[pd.Series, pd.DataFrame]:
    """
    Raw returns labeling method.

    This is the most basic and ubiquitous labeling method used as a precursor to almost any kind of financial data
    analysis or machine learning. User can specify simple or logarithmic returns, numerical or binary labels, a
    resample period, and whether returns are lagged to be forward looking.

    Parameters
    ----------
    prices : pd.Series or pd.DataFrame
        Time-indexed price data on stocks with which to calculate return.
    binary : bool
        If False, will return numerical returns. If True, will return the sign of the raw return.
    logarithmic : bool
        If False, will calculate simple returns. If True, will calculate logarithmic returns.
    resample_by : str or None
        If not None, the resampling period for price data prior to calculating returns. 'B' = per
        business day, 'W' = week, 'M' = month, etc. Will take the last observation for each period.
        For full details see `here.
        <https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects>`_
    lag : bool
        If True, returns will be lagged to make them forward-looking.

    Returns
    -------
    pd.Series or pd.DataFrame
        Raw returns on market data. User can specify whether returns will be based on
        simple or logarithmic return, and whether the output will be numerical or categorical.
    """
    # Apply resample, if applicable.
    if resample_by is not None:
        prices = prices.resample(resample_by).last()

    # Get return per period.
    if logarithmic:  # Log returns
        if lag:
            returns = np.log(prices).diff().shift(-1)
        else:
            returns = np.log(prices).diff()
    else:  # Simple returns
        if lag:
            returns = prices.pct_change(periods=1).shift(-1)
        else:
            returns = prices.pct_change(periods=1)

    # Return sign only if categorical labels desired.
    if binary:
        returns = returns.apply(np.sign)

    return returns
