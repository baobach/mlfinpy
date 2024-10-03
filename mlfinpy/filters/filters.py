"""
Filters are used to filter events based on some kind of trigger. For example a structural break
filter can be used to filter events where a structural break occurs. This event is then used to
measure the return from the event to some event horizon, say a day.
"""

from typing import Union

import numpy as np
import pandas as pd


# Snippet 2.4, page 39, The Symmetric CUSUM Filter.
def cusum_filter(
    raw_time_series: pd.Series, threshold: float, time_stamps: bool = True
) -> Union[pd.DatetimeIndex, list]:
    """
    The Symmetric Dynamic/Fixed CUSUM Filter.

    The CUSUM filter is a quality-control method, designed to detect a shift in the mean value of
    a measured quantity away from a target value. The filter is set up to identify a sequence of
    pside or downside divergences from any reset level zero. We sample a bar t if and only
    if S_t >= threshold, at which point S_t is reset to 0.

    One practical aspect that makes CUSUM filters appealing is that multiple events are not
    triggered by raw_time_series hovering around a threshold level, which is a flaw suffered
    by popular market signals such as Bollinger Bands. It will require a full run of length
    threshold for raw_time_series to trigger an event.

    Once we have obtained this subset of event-driven bars, we will let the ML algorithm determine
    whether the occurrence of such events constitutes actionable intelligence. Below is an implementation
    of the Symmetric CUSUM filter.

    Note: As per the book this filter is applied to closing prices but we extended
    it to also work on other time series such as volatility.

    Parameters
    ----------
    raw_time_series : pd.Series
        Close prices (or other time series, e.g. volatility).
    threshold : float
        When the abs(change) is larger than the threshold, the function captures.
    time_stamps : bool
        Default is to return a DateTimeIndex, change to false to have it return a list.

    Returns
    -------
    t_events : pd.DatetimeIndex or list
        Vector of datetimes when the events occurred. This is used later to sample.

    Notes
    -----
    Reference: Advances in Financial Machine Learning, Snippet 2.4, page 39.
    """

    t_events = []
    s_pos = 0
    s_neg = 0

    # log returns
    raw_time_series = pd.DataFrame(raw_time_series)  # Convert to DataFrame
    raw_time_series.columns = ["price"]
    raw_time_series["log_ret"] = raw_time_series.price.apply(np.log).diff()
    if isinstance(threshold, (float, int)):
        raw_time_series["threshold"] = threshold
    elif isinstance(threshold, pd.Series):
        raw_time_series.loc[threshold.index, "threshold"] = threshold
    else:
        raise ValueError("threshold is neither float nor pd.Series!")

    raw_time_series = raw_time_series.iloc[1:]  # Drop first na values

    # Get event time stamps for the entire series
    for tup in raw_time_series.itertuples():
        thresh = tup.threshold
        pos = float(s_pos + tup.log_ret)
        neg = float(s_neg + tup.log_ret)
        s_pos = max(0.0, pos)
        s_neg = min(0.0, neg)

        if s_neg < -thresh:
            s_neg = 0
            t_events.append(tup.Index)

        elif s_pos > thresh:
            s_pos = 0
            t_events.append(tup.Index)

    # Return DatetimeIndex or list
    if time_stamps:
        event_timestamps = pd.DatetimeIndex(t_events)
        return event_timestamps

    return t_events


def z_score_filter(
    raw_time_series: pd.Series,
    mean_window: int,
    std_window: int,
    z_score: float = 3,
    time_stamps: bool = True,
) -> Union[pd.DatetimeIndex, list]:
    """
    Filter which implements z_score filter.

    Parameters
    ----------
    raw_time_series : pd.Series
        Close prices (or other time series, e.g. volatility).
    mean_window : int
        Rolling mean window
    std_window : int
        Rolling std window
    z_score : float
        Number of standard deviations to trigger the event
    time_stamps : bool
        Default is to return a DateTimeIndex, change to false to have it return a list.

    Returns
    -------
    t_events : pd.DatetimeIndex or list
        Vector of datetimes when the events occurred. This is used later to sample.

    Notes
    -----
    Reference: Implement the idea of z-score filter here at
    [StackOverflow Question]
    (https://stackoverflow.com/questions/22583391/peak-signal-detection-in-realtime-timeseries-data).
    """
    t_events = raw_time_series[
        raw_time_series
        >= raw_time_series.rolling(window=mean_window).mean()
        + z_score * raw_time_series.rolling(window=std_window).std()
    ].index
    if time_stamps:
        event_timestamps = pd.DatetimeIndex(t_events)
        return event_timestamps
    return t_events
