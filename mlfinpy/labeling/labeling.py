from typing import List, Optional, Union

import numpy as np
import pandas as pd

from mlfinpy.util.multiprocess import mp_pandas_obj


# Snippet 3.2, page 45, Triple Barrier Labeling Method
def apply_pt_sl_on_t1(close: pd.Series, events: pd.Series, pt_sl: np.array, molecule: np.array) -> pd.DataFrame:
    # pragma: no cover
    """
    Triple Barrier Labeling Method

    This function applies the triple-barrier labeling method. It works on a set of datetime index values (molecule).
    This allows the program to parallelize the processing.Mainly it returns a DataFrame of timestamps
    regarding the time when the first barriers were reached.

    Parameters
    ----------
    close : pd.Series
        Close prices series.
    events : pd.Series
        Indices that signify "events" (see cusum_filter function for more details).
    pt_sl : np.array
        Element 0, indicates the profit taking level; Element 1 is stop loss level.
    molecule : np.array
        A set of datetime index values for processing.

    Returns
    -------
    out : pd.DataFrame
        Timestamps of when first barrier was touched

    Note
    ----
    Advances in Financial Machine Learning, Snippet 3.2, page 45.
    """
    # Apply stop loss/profit taking, if it takes place before t1 (end of event)
    events_ = events.loc[molecule]
    out = events_[["t1"]].copy(deep=True)

    profit_taking_multiple = pt_sl[0]
    stop_loss_multiple = pt_sl[1]

    # Profit taking active
    if profit_taking_multiple > 0:
        profit_taking = profit_taking_multiple * events_["trgt"]
    else:
        profit_taking = pd.Series(index=events.index)  # NaNs

    # Stop loss active
    if stop_loss_multiple > 0:
        stop_loss = -stop_loss_multiple * events_["trgt"]
    else:
        stop_loss = pd.Series(index=events.index)  # NaNs

    out["pt"] = pd.Series(dtype=events.index.dtype)
    out["sl"] = pd.Series(dtype=events.index.dtype)

    # Get events
    for loc, vertical_barrier in events_["t1"].fillna(close.index[-1]).items():
        closing_prices = close[loc:vertical_barrier]  # Path prices for a given trade
        cum_returns = (closing_prices / close[loc] - 1) * events_.at[loc, "side"]  # Path returns
        out.at[loc, "sl"] = cum_returns[cum_returns < stop_loss[loc]].index.min()  # Earliest stop loss date
        out.at[loc, "pt"] = cum_returns[cum_returns > profit_taking[loc]].index.min()  # Earliest profit taking date

    return out


# Snippet 3.4 page 49, Adding a Vertical Barrier
def add_vertical_barrier(t_events, close, num_days=0, num_hours=0, num_minutes=0, num_seconds=0):
    """
    Adding a Vertical Barrier

    For each index in `t_events`, it finds the timestamp of the next price bar at or immediately after
    a number of days num_days. This vertical barrier can be passed as an optional argument `t1` in `get_events`.

    This function creates a series that has all the timestamps of when the vertical barrier would be reached.

    Parameters
    ----------
    t_events : pd.Series
        Series of events timestamps from the filters e.g. Cusum filter, Z-score filter.
    close : pd.Series
        Close prices series.
    num_days : int, optional
        Number of days to add for vertical barrier.
    num_hours : int, optional
        Number of hours to add for vertical barrier.
    num_minutes : int, optional
        Number of minutes to add for vertical barrier.
    num_seconds : int, optional
        Number of seconds to add for vertical barrier.

    Returns
    -------
    verticle_barriers : pd.Series
        Timestamps of vertical barriers.

    Notes
    ------
    Advances in Financial Machine Learning, Snippet 3.4, page 49.
    """

    # Create a timedelta object based on the input parameters
    timedelta = pd.Timedelta(
        "{} days, {} hours, {} minutes, {} seconds".format(num_days, num_hours, num_minutes, num_seconds)
    )

    # Find index to closest to vertical barrier
    nearest_index = close.index.searchsorted(t_events + timedelta)

    # Exclude indexes which are outside the range of close price index
    nearest_index = nearest_index[nearest_index < close.shape[0]]

    # Find price index closest to vertical barrier time stamp
    nearest_timestamp = close.index[nearest_index]
    filtered_events = t_events[: nearest_index.shape[0]]

    # Create a series with the vertical barrier timestamps
    vertical_barriers = pd.Series(data=nearest_timestamp, index=filtered_events)

    return vertical_barriers


# Snippet 3.3 -> 3.6 page 50, Getting the Time of the First Touch, with Meta Labels
def get_events(
    close: pd.Series,
    t_events: pd.Series,
    pt_sl: List[float],
    target: pd.Series,
    min_ret: float,
    num_threads: int,
    vertical_barrier_times: Union[pd.Series, bool] = False,
    side_prediction: Optional[pd.Series] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Advances in Financial Machine Learning, Snippet 3.6 page 50.

    Getting the Time of the First Touch, with Meta Labels

    This function is orchestrator to meta-label the data, in conjunction with the Triple Barrier Method.

    Parameters
    ----------
    close : pd.Series
        Close prices
    t_events : pd.Series
        of t_events. These are timestamps that will seed every triple barrier.
        These are the timestamps selected by the sampling procedures discussed in Chapter 2, Section 2.5.
        Eg: CUSUM Filter
    pt_sl : List[float]
        Element 0, indicates the profit taking level; Element 1 is stop loss level.
        A non-negative float that sets the width of the two barriers. A 0 value means that the respective
        horizontal barrier (profit taking and/or stop loss) will be disabled.
    target : pd.Series
        of values that are used (in conjunction with pt_sl) to determine the width
        of the barrier. In this program this is daily volatility series.
    min_ret : float
        The minimum target return required for running a triple barrier search.
    num_threads : int
        The number of threads concurrently used by the function.
    vertical_barrier_times : Union[pd.Series, bool]
        A pandas series with the timestamps of the vertical barriers.
        We pass a False when we want to disable vertical barriers.
    side_prediction : Optional[pd.Series]
        Side of the bet (long/short) as decided by the primary model
    verbose : bool
        Flag to report progress on asynch jobs

    Returns
    -------
    events : pd.DataFrame
        Dataframe of first touch events with meta-labels.
        - events.index is event's starttime
        - events['t1'] is event's endtime
        - events['trgt'] is event's target
        - events['side'] (optional) implies the algo's position side
        - events['pt'] is profit taking multiple
        - events['sl'] is stop loss multiple
    """
    # 1) Get target
    target = target.reindex(t_events)
    target = target[target > min_ret]  # min_ret

    # 2) Get vertical barrier (max holding period)
    if vertical_barrier_times is False:
        vertical_barrier_times = pd.Series(pd.NaT, index=t_events, dtype=t_events.dtype)

    # 3) Form events object, apply stop loss on vertical barrier
    if side_prediction is None:
        side_ = pd.Series(1.0, index=target.index)
        pt_sl_ = [pt_sl[0], pt_sl[0]]
    else:
        side_ = side_prediction.reindex(target.index)  # Subset side_prediction on target index.
        pt_sl_ = pt_sl[:2]

    # Create a new df with [v_barrier, target, side] and drop rows that are NA in target
    events = pd.concat({"t1": vertical_barrier_times, "trgt": target, "side": side_}, axis=1)
    events = events.dropna(subset=["trgt"])

    # Apply Triple Barrier
    first_touch_dates = mp_pandas_obj(
        func=apply_pt_sl_on_t1,
        pd_obj=("molecule", events.index),
        num_threads=num_threads,
        close=close,
        events=events,
        pt_sl=pt_sl_,
        verbose=verbose,
    )

    for ind in events.index:
        events.at[ind, "t1"] = first_touch_dates.loc[ind, :].dropna().min()

    if side_prediction is None:
        events = events.drop("side", axis=1)

    # Add profit taking and stop loss multiples for vertical barrier calculations
    events["pt"] = pt_sl[0]
    events["sl"] = pt_sl[1]

    return events


# Snippet 3.9, pg 55, Question 3.3
def barrier_touched(out_df: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """
    Adjust the getBins function (Snippet 3.7) to return a 0 whenever the vertical barrier is the one touched first.

    Top horizontal barrier: 1
    Bottom horizontal barrier: -1
    Vertical barrier: 0

    Parameters
    ----------
    out_df : pd.DataFrame
        Returns and target.
    events : pd.DataFrame
        The original events data frame. Contains the pt sl multiples needed here.

    Returns
    -------
    pd.DataFrame
        Returns, target, and labels.

    Notes
    -----
    Advances in Financial Machine Learning, Snippet 3.9, page 55, Question 3.3.
    """
    store = []
    for date_time, values in out_df.iterrows():
        ret = values["ret"]
        target = values["trgt"]

        pt_level_reached = ret > np.log(1 + target) * events.loc[date_time, "pt"]
        sl_level_reached = ret < -np.log(1 + target) * events.loc[date_time, "sl"]

        if ret > 0.0 and pt_level_reached:
            # Top barrier reached
            store.append(1)
        elif ret < 0.0 and sl_level_reached:
            # Bottom barrier reached
            store.append(-1)
        else:
            # Vertical barrier reached
            store.append(0)

    # Save to 'bin' column and return
    out_df["bin"] = store
    return out_df


# Snippet 3.4 -> 3.7, page 51, Labeling for Side & Size with Meta Labels
def get_bins(triple_barrier_events: pd.DataFrame, close: pd.Series) -> pd.DataFrame:
    """
    Labeling for Side & Size with Meta Labels

    Compute event's outcome (including side information, if provided).
    events is a DataFrame where:

    Now the possible values for labels in out['bin'] are {0,1}, as opposed to whether to take the bet or pass,
    a purely binary prediction. When the predicted label the previous feasible values {−1,0,1}.
    The ML algorithm will be trained to decide is 1, we can use the probability of this secondary prediction
    to derive the size of the bet, where the side (sign) of the position has been set by the primary model.

    Parameters
    ----------
    triple_barrier_events : pd.DataFrame
        DataFrame returned by 'get_events' with columns:
        - index: event starttime
        - vertical_barriers: event endtime
        - trgt: event target
        - side (optional): position side
        Case 1: ('side' not in events): bin in (-1,1) <-label by price action.
        Case 2: ('side' in events): bin in (0,1) <-label by pnl (meta-labeling).
    close : pd.Series
        Close prices series.

    Returns
    -------
    out_df : pd.DataFrame
        Meta-labeled events.

    Notes
    -----
    Advances in Financial Machine Learning, Snippet 3.7, page 51.
    """

    # 1) Align prices with their respective events
    events_ = triple_barrier_events.dropna(subset=["t1"])
    all_dates = events_.index.union(other=events_["t1"].array).drop_duplicates()
    prices = close.reindex(all_dates, method="bfill")

    # 2) Create out DataFrame
    out_df = pd.DataFrame(index=events_.index)
    # Need to take the log returns, else your results will be skewed for short positions
    out_df["ret"] = np.log(prices.loc[events_["t1"].array].array) - np.log(prices.loc[events_.index])
    out_df["trgt"] = events_["trgt"]

    # Meta labeling: Events that were correct will have pos returns
    if "side" in events_:
        out_df["ret"] = out_df["ret"] * events_["side"]  # meta-labeling

    # Added code: label 0 when vertical barrier reached
    out_df = barrier_touched(out_df, triple_barrier_events)

    # Meta labeling: label incorrect events with a 0
    if "side" in events_:
        out_df.loc[out_df["ret"] <= 0, "bin"] = 0

    # Transform the log returns back to normal returns.
    out_df["ret"] = np.exp(out_df["ret"]) - 1

    # Add the side to the output. This is useful for when a meta label model must be fit
    tb_cols = triple_barrier_events.columns
    if "side" in tb_cols:
        out_df["side"] = triple_barrier_events["side"]

    return out_df


# Snippet 3.8 page 54
def drop_labels(events: pd.DataFrame, min_pct: float = 0.05) -> pd.DataFrame:
    """
    This function recursively eliminates rare observations.

    Parameters
    ----------
    events : pd.DataFrame
        Events.
    min_pct : float, optional
        A fraction used to decide if the observation occurs less than that fraction.
        Defaults to .05.

    Returns
    -------
    pd.DataFrame
        Events.

    Notes
    -----
    Advances in Financial Machine Learning, Snippet 3.8 page 54.
    """
    # Apply weights, drop labels with insufficient examples
    while True:
        df0 = events["bin"].value_counts(normalize=True)

        if df0.min() > min_pct or df0.shape[0] < 3:
            break

        print("dropped label: ", df0.idxmin(), df0.min())
        events = events[events["bin"] != df0.idxmin()]

    return events
