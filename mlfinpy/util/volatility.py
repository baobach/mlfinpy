"""
This module contain a collection of volatility estimators.
"""

import numpy as np
import pandas as pd

# pylint: disable=redefined-builtin


def get_daily_vol(close: pd.Series, lookback: int = 100) -> pd.Series:
    """
    Daily Volatility Estimates
    Computes the daily volatility at intraday estimation points.
    In practice we want to set profit taking and stop-loss limits that are a function of the risks involved
    in a bet. Otherwise, sometimes we will be aiming too high (tao ≫ sigma_t_i,0), and sometimes too low
    (tao ≪ sigma_t_i,0 ), considering the prevailing volatility. Snippet 3.1 computes the daily volatility
    at intraday estimation points, applying a span of lookback days to an exponentially weighted moving
    standard deviation.

    Parameters
    ----------
    close : pd.Series
        Closing prices series.
    lookback : int, default value = 100
        This value is used to compute alpha decay.
        value in the ewm() method:
        `alpha` = 2 / (`span` + 1) for `span` => 1.

    Returns
    -------
    daily_vols : pd.Series
    A pandas series of daily volatility compute as a standard

    Note
    ----
    Advances in Financial Machine Learning, Snippet 3.1, page 44.
    This function is used to compute dynamic thresholds for profit taking and stop loss limits.
    See the pandas documentation for details on the pandas.Series.ewm function.
    """
    # daily vol re-indexed to close
    df0 = close.index.searchsorted(close.index - pd.Timedelta(days=1))
    df0 = df0[df0 > 0]
    df0 = pd.Series(close.index[df0 - 1], index=close.index[close.shape[0] - df0.shape[0] :])

    df0 = close.loc[df0.index] / close.loc[df0.array].array - 1  # daily returns
    df0 = df0.ewm(span=lookback).std()
    return df0


def get_parksinson_vol(high: pd.Series, low: pd.Series, window: int = 20) -> pd.Series:
    """
    Parkinson volatility estimator

    Parameters
    ----------
    high : pd.Series
        High prices.
    low : pd.Series
        Low prices.
    window : int, default value = 20
        Window used for estimation.

    Returns
    -------
    pd.Series
        Parkinson volatility.
    """
    ret = np.log(high / low)  # High/Low return
    estimator = 1 / (4 * np.log(2)) * (ret**2)
    return np.sqrt(estimator.rolling(window=window).mean())


def get_garman_class_vol(
    open: pd.Series,  # Open prices
    high: pd.Series,  # High prices
    low: pd.Series,  # Low prices
    close: pd.Series,  # Close prices
    window: int = 20,  # Window used for estimation
) -> pd.Series:
    """
    Garman-Class volatility estimator

    Parameters
    ----------
    open : pd.Series
        Open prices.
    high : pd.Series
        High prices.
    low : pd.Series
        Low prices.
    close : pd.Series
        Close prices
    window : int, default 20
        Window used for estimation.

    Returns
    -------
    pd.Series
        Garman-Class volatility.
    """
    ret = np.log(high / low)  # High/Low return
    close_open_ret = np.log(close / open)  # Close/Open return
    estimator = 0.5 * ret**2 - (2 * np.log(2) - 1) * close_open_ret**2
    return np.sqrt(estimator.rolling(window=window).mean())


def get_yang_zhang_vol(
    open: pd.Series,  # Open prices
    high: pd.Series,  # High prices
    low: pd.Series,  # Low prices
    close: pd.Series,  # Close prices
    window: int = 20,  # Window used for estimation
) -> pd.Series:
    """
    Yang-Zhang volatility estimator

    Parameters
    ----------
    open : pd.Series
        Open prices.
    high : pd.Series
        High prices.
    low : pd.Series
        Low prices.
    close : pd.Series
        Close prices.
    window : int, default 20
        Window used for estimation.

    Returns
    -------
    pd.Series
        Yang-Zhang volatility.
    """
    k = 0.34 / (1.34 + (window + 1) / (window - 1))

    open_prev_close_ret = np.log(open / close.shift(1))
    close_prev_open_ret = np.log(close / open.shift(1))

    high_close_ret = np.log(high / close)
    high_open_ret = np.log(high / open)
    low_close_ret = np.log(low / close)
    low_open_ret = np.log(low / open)

    sigma_open_sq = 1 / (window - 1) * (open_prev_close_ret**2).rolling(window=window).sum()
    sigma_close_sq = 1 / (window - 1) * (close_prev_open_ret**2).rolling(window=window).sum()
    sigma_rs_sq = (
        1 / (window - 1) * (high_close_ret * high_open_ret + low_close_ret * low_open_ret).rolling(window=window).sum()
    )

    return np.sqrt(sigma_open_sq + k * sigma_close_sq + (1 - k) * sigma_rs_sq)
