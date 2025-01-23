"""
This module contains functionality for determining bet sizes for investments based on machine learning predictions.
These implementations are based on bet sizing approaches described in Chapter 10.
"""

import numpy as np
import pandas as pd
from scipy.stats import moment, norm

from mlfinpy.bet_sizing.ch10_snippets import (
    avg_active_signals,
    bet_size,
    discrete_signal,
    get_signal,
    get_target_pos,
    get_w,
    limit_price,
)
from mlfinpy.bet_sizing.ef3m import M2N, most_likely_parameters, raw_moment


def bet_size_probability(events, prob, num_classes, pred=None, step_size=0.0, average_active=False, num_threads=1):
    """
    Calculates the bet size using the predicted probability. Note that if 'average_active' is True, the returned
    pandas.Series will be twice the length of the original since the average is calculated at each bet's open and close.

    Parameters
    ----------
    events : pd.DataFrame
        Contains at least the column 't1', the expiry datetime of the product, with a datetime index, the datetime the
        position was taken.
    prob : pd.Series
        The predicted probability.
    num_classes : int
        The number of predicted bet sides.
    pred : pd.Series, optional
        The predicted bet side. Default value is None which will return a relative bet size (i.e. without multiplying
        by the side).
    step_size : float, optional
        The step size at which the bet size is discretized, default is 0.0 which imposes no discretization.
    average_active : bool, optional
        Option to average the size of active bets, default value is False.
    num_threads : int, optional
        The number of processing threads to utilize for multiprocessing, default value is 1.

    Returns
    -------
    pd.Series
        The bet size, with the time index.
    """
    signal_0 = get_signal(prob, num_classes, pred)
    events_0 = signal_0.to_frame("signal").join(events["t1"], how="left")
    if average_active:
        signal_1 = avg_active_signals(events_0, num_threads)
    else:
        signal_1 = events_0.signal

    if abs(step_size) > 0:
        signal_1 = discrete_signal(signal0=signal_1, step_size=abs(step_size))

    return signal_1


def bet_size_dynamic(
    current_pos, max_pos, market_price, forecast_price, cal_divergence=10, cal_bet_size=0.95, func="sigmoid"
):
    """
    Calculates the bet sizes, target position, and limit price as the market price and forecast price fluctuate.

    Parameters
    ----------
    current_pos : pd.Series or int
        Current position.
    max_pos : pd.Series or int
        Maximum position.
    market_price : pd.Series or float
        Market price.
    forecast_price : pd.Series or float
        Forecast price.
    cal_divergence : float, optional
        The divergence to use in calibration.
    cal_bet_size : float, optional
        The bet size to use in calibration.
    func : str, optional
        Function to use for dynamic calculation. Valid options are: 'sigmoid', 'power'.

    Returns
    -------
    pd.DataFrame
        Bet size (bet_size), target position (t_pos), and limit price (l_p).
    """
    # Create a dictionary of bet size variables for easier handling.
    d_vars = {"pos": current_pos, "max_pos": max_pos, "m_p": market_price, "f": forecast_price}
    events_0 = confirm_and_cast_to_df(d_vars)

    # Calibrate w.
    w_param = get_w(cal_divergence, cal_bet_size, func)
    # Compute the target bet position.
    events_0["t_pos"] = events_0.apply(lambda x: get_target_pos(w_param, x.f, x.m_p, x.max_pos, func), axis=1)
    # Compute the break even limit price.
    events_0["l_p"] = events_0.apply(lambda x: limit_price(x.t_pos, x.pos, x.f, w_param, x.max_pos, func), axis=1)
    # Compute the bet size.
    events_0["bet_size"] = events_0.apply(lambda x: bet_size(w_param, x.f - x.m_p, func), axis=1)

    return events_0[["bet_size", "t_pos", "l_p"]]


def bet_size_budget(events_t1, sides):
    """
    Calculates a bet size from the bet sides and start and end times.

    Parameters
    ----------
    events_t1 : pd.Series
        The end datetime of the position with the start datetime as the index.
    sides : pd.Series
        The side of the bet with the start datetime as index. Index must match the 'events_t1'
        argument exactly. Bet sides less than zero are interpreted as short, bet sides greater
        than zero are interpreted as long.

    Returns
    -------
    pd.DataFrame
        The 'events_t1' and 'sides' arguments as columns, with the number of concurrent active
        long and short bets, as well as the bet size, in additional columns.
    """
    events_1 = get_concurrent_sides(events_t1, sides)
    active_long_max, active_short_max = events_1["active_long"].max(), events_1["active_short"].max()
    frac_active_long = events_1["active_long"] / active_long_max if active_long_max > 0 else 0
    frac_active_short = events_1["active_short"] / active_short_max if active_short_max > 0 else 0
    events_1["bet_size"] = frac_active_long - frac_active_short

    return events_1


def bet_size_reserve(
    events_t1,
    sides,
    fit_runs=100,
    epsilon=1e-5,
    factor=5,
    variant=2,
    max_iter=10_000,
    num_workers=1,
    return_parameters=False,
):
    """
    Calculates the bet size from bet sides and start and end times.

    Parameters
    ----------
    events_t1 : pd.Series
        The end datetime of the position with the start datetime as the index.
    sides : pd.Series
        The side of the bet with the start datetime as index. Index must match
        the 'events_t1' argument exactly. Bet sides less than zero are interpreted
        as short, bet sides greater than zero are interpreted as long.
    fit_runs : int, optional
        Number of runs to execute when trying to fit the distribution.
    epsilon : float, optional
        Error tolerance.
    factor : float, optional
        Lambda factor from equations.
    variant : int, optional
        Which algorithm variant to use, 1 or 2.
    max_iter : int, optional
        Maximum number of iterations after which to terminate loop.
    num_workers : int, optional
        Number of CPU cores to use for multiprocessing execution, set to -1 to use
        all CPU cores. Default is 1.
    return_parameters : bool, optional
        If True, function also returns a dictionary of the fitted mixture parameters.

    Returns
    -------
    pd.DataFrame
        The 'events_t1' and 'sides' arguments as columns, with the number of concurrent active long,
        short bets, the difference between long and short, and the bet size in additional columns.
    dict, optional
        The mixture parameters if 'return_parameters' is set to True.
    """
    events_active = get_concurrent_sides(events_t1, sides)
    # Calculate the concurrent difference in active bets: c_t = <current active long> - <current active short>
    events_active["c_t"] = events_active["active_long"] - events_active["active_short"]
    # Calculate the first 5 centered and raw moments from the c_t distribution.
    central_mmnts = [moment(events_active["c_t"].to_numpy(), moment=i) for i in range(1, 6)]
    raw_mmnts = raw_moment(central_moments=central_mmnts, dist_mean=events_active["c_t"].mean())
    # Fit the mixture of distributions.
    m2n = M2N(
        raw_mmnts,
        epsilon=epsilon,
        factor=factor,
        n_runs=fit_runs,
        variant=variant,
        max_iter=max_iter,
        num_workers=num_workers,
    )
    df_fit_results = m2n.mp_fit()
    fit_params = most_likely_parameters(df_fit_results)
    params_list = [fit_params[key] for key in ["mu_1", "mu_2", "sigma_1", "sigma_2", "p_1"]]
    # Calculate the bet size.
    events_active["bet_size"] = events_active["c_t"].apply(lambda c: single_bet_size_mixed(c, params_list))

    if return_parameters:
        return events_active, fit_params
    return events_active


def confirm_and_cast_to_df(d_vars):
    """
    Accepts either pandas.Series (with a common index) or integer/float values, casts all non-pandas.Series values
    to Series, and returns a pandas.DataFrame for further calculations.

    Parameters
    ----------
    d_vars : dict
        A dictionary where the values are either pandas.Series or single int/float values. All pandas.Series passed
        are assumed to have the same index. The keys of the dictionary will be used for column names in the
        returned pandas.DataFrame.

    Returns
    -------
    pd.DataFrame
        The values from the input dictionary in pandas.DataFrame format, with dictionary keys as column names.
    """
    any_series = False  # Are any variables a pandas.Series?
    all_series = True  # Are all variables a pandas.Series?
    ser_len = 0
    for var in d_vars.values():
        any_series = any_series or isinstance(var, pd.Series)
        all_series = all_series and isinstance(var, pd.Series)

        if isinstance(var, pd.Series):
            ser_len = var.size
            idx = var.index

    # Handle data types if there are no pandas.Series variables.
    if not any_series:
        for k in d_vars:
            d_vars[k] = pd.Series(data=[d_vars[k]], index=[0])

    # Handle data types if some but not all variables are pandas.Series.
    if any_series and not all_series:
        for k in d_vars:
            if not isinstance(d_vars[k], pd.Series):
                d_vars[k] = pd.Series(data=np.array([d_vars[k] for i in range(ser_len)]), index=idx)

    # Combine Series to form a DataFrame.
    events = pd.concat(list(d_vars.values()), axis=1)
    events.columns = list(d_vars.keys())

    return events


def get_concurrent_sides(events_t1, sides):
    """
    Given the side of the position along with its start and end timestamps, this function returns two pandas.Series
    indicating the number of concurrent long and short bets at each timestamp.

    Parameters
    ----------
    events_t1 : pd.Series
        The end datetime of the position with the start datetime as the index.
    sides : pd.Series
        The side of the bet with the start datetime as index. Index must match the 'events_t1' argument exactly.
        Bet sides less than zero are interpreted as short, bet sides greater than zero are interpreted as long.

    Returns
    -------
    pd.DataFrame
        The 'events_t1' and 'sides' arguments as columns, with two additional columns indicating the number of
        concurrent active long and active short bets at each timestamp.
    """
    events_0 = pd.DataFrame({"t1": events_t1, "side": sides})
    events_0["active_long"] = 0
    events_0["active_short"] = 0

    for idx in events_0.index:
        # A bet side greater than zero indicates a long position.
        df_long_active_idx = set(
            events_0[(events_0.index <= idx) & (events_0["t1"] > idx) & (events_0["side"] > 0)].index
        )
        events_0.loc[idx, "active_long"] = len(df_long_active_idx)
        # A bet side less than zero indicates a short position.
        df_short_active_idx = set(
            events_0[(events_0.index <= idx) & (events_0["t1"] > idx) & (events_0["side"] < 0)].index
        )
        events_0.loc[idx, "active_short"] = len(df_short_active_idx)

    return events_0


def cdf_mixture(x_val, parameters):
    """
    The cumulative distribution function of a mixture of 2 normal distributions, evaluated at x_val.

    Parameters
    ----------
    x_val : float
        Value at which to evaluate the CDF.
    parameters : list
        The parameters of the mixture, [mu_1, mu_2, sigma_1, sigma_2, p_1].

    Returns
    -------
    float
        CDF of the mixture.
    """
    mu_1, mu_2, sigma_1, sigma_2, p_1 = parameters  # Parameters reassigned for clarity.
    return p_1 * norm.cdf(x_val, mu_1, sigma_1) + (1 - p_1) * norm.cdf(x_val, mu_2, sigma_2)


def single_bet_size_mixed(c_t, parameters):
    """
    Returns the single bet size based on the description provided in question 10.4(c), provided the difference in
    concurrent long and short positions, c_t, and the fitted parameters of the mixture of two Gaussian distributions.

    Parameters
    ----------
    c_t : int
        The difference in the number of concurrent long bets minus short bets.
    parameters : list
        The parameters of the mixture, [mu_1, mu_2, sigma_1, sigma_2, p_1].

    Returns
    -------
    float
        Bet size.
    """
    if c_t >= 0:
        single_bet_size = (cdf_mixture(c_t, parameters) - cdf_mixture(0, parameters)) / (1 - cdf_mixture(0, parameters))
    else:
        single_bet_size = (cdf_mixture(c_t, parameters) - cdf_mixture(0, parameters)) / cdf_mixture(0, parameters)
    return single_bet_size
