"""
Advances in Financial Machine Learning, Marcos Lopez de Prado
Chapter 2: Financial Data Structures

This module contains the functions to help users create structured financial data from raw unstructured data,
in the form of time, tick, volume, and dollar bars.

These bars are used throughout the text book (Advances in Financial Machine Learning, By Marcos Lopez de Prado, 2018,
pg 25) to build the more interesting features for predicting financial time series data.

These financial data structures have better statistical properties when compared to those based on fixed time interval
sampling. A great paper to read more about this is titled: The Volume Clock: Insights into the high frequency paradigm,
Lopez de Prado, et al.

Many of the projects going forward will require Dollar and Volume bars.
"""

# Imports
from typing import Union, Iterable, Optional

import numpy as np
import pandas as pd

from mlfinpy.data_structure.base_bars import BaseBars


class StandardBars(BaseBars):
    """
    Encapsulates the logic for constructing the standard bars from Chapter 2 of "Advances in Financial Machine Learning" by Marcos Lopez de Prado. This class is not intended for direct use. Instead, utilize package functions like `get_dollar_bars` to create an instance and construct the standard bars.
    """

    def __init__(
        self, inform_bar_type: str, threshold: int = 50000, batch_size: int = 20000000
    ):
        """
        Constructor for Standard Bars

        Parameters
        ----------
        inform_bar_type : str
            Type of standard bar to create. Example: `dollar_run`, `volume_imbalance`.
        threshold : int
            Threshold interm of dollar value, traded volume, or ticks.
        batch_size : int
            Number of rows to read in from the csv, per batch (default is 2e7).
        """
        BaseBars.__init__(self, inform_bar_type, batch_size)

        # Threshold at which to sample
        self.threshold = threshold

    def _reset_cache(self):
        """
        Implementation of abstract method `_reset_cache` for standard bars.
        """
        self.open_price = None
        self.high_price, self.low_price = -np.inf, np.inf
        self.cum_statistics = {
            "cum_ticks": 0,
            "cum_dollar_value": 0,
            "cum_volume": 0,
            "cum_buy_volume": 0,
        }

    def _extract_bars(self, raw_tick_data: Union[list, tuple, np.ndarray]) -> list:
        """
        For loop which compiles the various bars: dollar, volume, or tick.

        Parameters
        ----------
        raw_tick_data : list or tuple or np.ndarray
            Contains 3 columns - 'date_time', 'price', and 'volume'.

        Returns
        -------
        list
            Bars built using the current batch.
        """

        # Iterate over rows
        list_bars = []

        for row in raw_tick_data:
            # Set variables
            date_time = row[0]
            self.tick_num += 1
            price = float(row[1])
            volume = row[2]
            dollar_value = price * volume
            signed_tick = self._apply_tick_rule(price)

            if self.open_price is None:
                self.open_price = price

            # Update high low prices
            self.high_price, self.low_price = self._update_high_low(price)

            # Calculations
            self.cum_statistics["cum_ticks"] += 1
            self.cum_statistics["cum_dollar_value"] += dollar_value
            self.cum_statistics["cum_volume"] += volume
            if signed_tick == 1:
                self.cum_statistics["cum_buy_volume"] += volume

            # If threshold reached then take a sample
            if (
                self.cum_statistics[self.inform_bar_type] >= self.threshold
            ):  # pylint: disable=eval-used
                self._create_bars(
                    date_time, price, self.high_price, self.low_price, list_bars
                )

                # Reset cache
                self._reset_cache()
        return list_bars


def get_dollar_bars(
    file_path_or_df: Union[str, Iterable[str], pd.DataFrame],
    threshold: float = 70000000,
    batch_size: int = 20000000,
    verbose: bool = True,
    to_csv: bool = False,
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Creates a DataFrame of dollar bars with columns: date_time, open, high, low, close, volume, cum_buy_volume, cum_ticks, cum_dollar_value.

    Parameters
    ----------
    file_path_or_df : str or iterable of str or pd.DataFrame
        Path to the csv file(s) or Pandas Data Frame containing raw tick data in the format[date_time, price, volume].
    threshold : float
        A cumulative traded dollar value above this threshold triggers a sample to be taken.
    batch_size : int
        The number of rows per batch. Less RAM = smaller batch size.
    verbose : bool
        Print out batch numbers.
    to_csv : bool
        Save bars to csv after every batch run.
    output_path : str
        Path to csv file, if to_csv is True.

    Returns
    -------
    pd.DataFrame
        Dataframe of dollar bars.

    Notes
    -----
    Following the paper "The Volume Clock: Insights into the high frequency paradigm" by Lopez de Prado, et al, it is suggested that using 1/50 of the average daily dollar value, would result in more desirable statistical properties.
    """

    bars = StandardBars(
        inform_bar_type="cum_dollar_value", threshold=threshold, batch_size=batch_size
    )
    dollar_bars = bars.batch_run(
        file_path_or_df=file_path_or_df,
        verbose=verbose,
        to_csv=to_csv,
        output_path=output_path,
    )
    return dollar_bars


def get_volume_bars(
    file_path_or_df: Union[str, Iterable[str], pd.DataFrame],
    threshold: float = 70000000,
    batch_size: int = 20000000,
    verbose: bool = True,
    to_csv: bool = False,
    output_path: Optional[str] = None,
):
    """
    Create a DataFrame of volume bars with columns: date_time, open, high, low, close, volume, cum_buy_volume, cum_ticks, cum_dollar_value.

    Parameters
    ----------
    file_path_or_df : str or iterable of str or pd.DataFrame
        Path to the csv file(s) or Pandas Data Frame containing raw tick data in the format[date_time, price, volume]
    threshold : float
        A cumulative traded volume above this threshold triggers a sample to be taken.
    batch_size : int
        The number of rows per batch. Less RAM = smaller batch size.
    verbose : bool
        Print out batch numbers.
    to_csv : bool
        Save bars to csv after every batch run.
    output_path : str
        Path to csv file, if to_csv is True.

    Returns
    -------
    pd.DataFrame
        Dataframe of volume bars.

    Notes
    -----
    Following the paper "The Volume Clock: Insights into the high frequency paradigm" by Lopez de Prado, et al, it is suggested that using 1/50 of the average daily volume, would result in more desirable statistical properties.
    """
    bars = StandardBars(
        inform_bar_type="cum_volume", threshold=threshold, batch_size=batch_size
    )
    volume_bars = bars.batch_run(
        file_path_or_df=file_path_or_df,
        verbose=verbose,
        to_csv=to_csv,
        output_path=output_path,
    )
    return volume_bars


def get_tick_bars(
    file_path_or_df: Union[str, Iterable[str], pd.DataFrame],
    threshold: float = 70000000,
    batch_size: int = 20000000,
    verbose: bool = True,
    to_csv: bool = False,
    output_path: Optional[str] = None,
):
    """
    Create a DataFrame of tick bars with columns: date_time, open, high, low, close, volume, cum_buy_volume, cum_ticks, cum_dollar_value.

    Parameters
    ----------
    file_path_or_df : str or iterable of str or pd.DataFrame
        Path to the csv file(s) or Pandas Data Frame containing raw tick data in the format[date_time, price, volume]
    threshold : float
        A cumulative number of ticks above this threshold triggers a sample to be taken.
    batch_size : int
        The number of rows per batch. Less RAM = smaller batch size.
    verbose : bool
        Print out batch numbers.
    to_csv : bool
        Save bars to csv after every batch run.
    output_path : str
        Path to csv file, if to_csv is True.

    Returns
    -------
    pd.DataFrame
        Dataframe of tick bars.

    Notes
    -----
    Following the paper "The Volume Clock: Insights into the high frequency paradigm" by Lopez de Prado, et al, it is suggested that using 1/50 of the average daily volume, would result in more desirable statistical properties.
    """
    bars = StandardBars(
        inform_bar_type="cum_ticks", threshold=threshold, batch_size=batch_size
    )
    tick_bars = bars.batch_run(
        file_path_or_df=file_path_or_df,
        verbose=verbose,
        to_csv=to_csv,
        output_path=output_path,
    )
    return tick_bars
