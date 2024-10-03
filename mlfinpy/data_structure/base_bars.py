"""
Base class for various bar types.

Provides shared logic to minimize duplicated code across bar type implementations.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Union, Generator, Iterable, Optional

import numpy as np
import pandas as pd

from mlfinpy.util.fast_ewma import ewma
from mlfinpy.util.helper import crop_data_frame_in_batches


class BaseBars(ABC):
    """
    Abstract base class for financial data structures, providing a shared structure for standard and information-driven bars.
    This class contains common attributes and methods for bar types, including those specific to information bars.
    """

    def __init__(self, inform_bar_type: str, batch_size: int = 2e7):
        """
        Constructor

        Parameters
        ----------
        inform_bar_type : str
            Type of imbalance bar to create. Example: dollar_imbalance.
        batch_size : int, optional
            Number of rows to read in from the csv, per batch (default is 2e7).
        """
        # Base properties
        self.inform_bar_type = inform_bar_type
        self.batch_size = batch_size
        self.prev_tick_rule = 0

        # Cache properties
        self.open_price, self.prev_price, self.close_price = None, None, None
        self.high_price, self.low_price = -np.inf, np.inf
        self.cum_statistics = {
            "cum_ticks": 0,
            "cum_dollar_value": 0,
            "cum_volume": 0,
            "cum_buy_volume": 0,
        }
        self.tick_num = 0  # Tick number when bar was formed

        # Batch_run properties
        self.flag = (
            False  # The first flag is false since the first batch doesn't use the cache
        )

    def batch_run(
        self,
        file_path_or_df: Union[str, Iterable[str], pd.DataFrame],
        verbose: bool = True,
        to_csv: bool = False,
        output_path: Optional[str] = None,
    ) -> Union[pd.DataFrame, None]:
        """
        Reads csv file(s) or pd.DataFrame in batches and constructs the financial data structure in the form of a DataFrame.

        Parameters
        ----------
        file_path_or_df : str, iterable of str, or pd.DataFrame
            Path to the csv file(s) or Pandas Data Frame containing raw tick data in the format [date_time, price, volume]
        verbose : bool, optional
            Flag whether to print message on each processed batch or not (default is True)
        to_csv : bool, optional
            Flag for writing the results of bars generation to local csv file, or to in-memory DataFrame (default is False)
        output_path : str, optional
            Path to results file, if to_csv is True (default is None)

        Returns
        -------
        pd.DataFrame or None
            Financial data structure

        Notes
        -----
        The csv file or DataFrame must have only 3 columns: date_time, price, & volume.
        """

        if to_csv is True:
            header = (
                True  # if to_csv is True, header should written on the first batch only
            )
            open(output_path, "w").close()  # clean output csv file

        if verbose:  # pragma: no cover
            print("Reading data in batches:")

        # Read csv in batches
        count = 0
        final_bars = []
        cols = [
            "date_time",
            "tick_num",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "cum_buy_volume",
            "cum_ticks",
            "cum_dollar_value",
        ]
        for batch in self._batch_iterator(file_path_or_df):
            if verbose:  # pragma: no cover
                print("Batch number:", count)

            list_bars = self.run(raw_tick_data=batch)

            if to_csv is True:
                pd.DataFrame(list_bars, columns=cols).to_csv(
                    output_path, header=header, index=False, mode="a"
                )
                header = False
            else:
                # Append to bars list
                final_bars += list_bars
            count += 1

        if verbose:  # pragma: no cover
            print("Returning bars \n")

        # Return a DataFrame
        if final_bars:
            bars_df = pd.DataFrame(final_bars, columns=cols)
            return bars_df

        # Processed DataFrame is stored in .csv file, return None
        return None

    def _batch_iterator(
        self, file_path_or_df: Union[str, Iterable[str], pd.DataFrame]
    ) -> Generator[pd.DataFrame, None, None]:
        """
        Yields batches of data from the provided file path or DataFrame.

        Parameters
        ----------
        file_path_or_df : str, iterable of str, or pd.DataFrame
            Path to the csv file(s) or Pandas Data Frame containing raw tick data.

        Yields
        ------
        pd.DataFrame
            A batch of data read from the file or DataFrame.

        Notes
        -----
        This method is used to iterate over the data in batches, allowing for more efficient processing.
        """
        if isinstance(file_path_or_df, (list, tuple)):
            # Assert format of all files
            for file_path in file_path_or_df:
                self._read_first_row(file_path)
            for file_path in file_path_or_df:
                for batch in pd.read_csv(
                    file_path, chunksize=self.batch_size, parse_dates=[0]
                ):
                    yield batch

        elif isinstance(file_path_or_df, str):
            self._read_first_row(file_path_or_df)
            for batch in pd.read_csv(
                file_path_or_df, chunksize=self.batch_size, parse_dates=[0]
            ):
                yield batch

        elif isinstance(file_path_or_df, pd.DataFrame):
            for batch in crop_data_frame_in_batches(file_path_or_df, self.batch_size):
                yield batch

        else:
            raise ValueError(
                "Invalid input type for file_path_or_df. Expected a string (path to a csv file), an iterable of strings, or a pd.DataFrame."
            )

    def _read_first_row(self, file_path: str):
        """
        Reads the first row of a CSV file and checks its format.

        Parameters
        ----------
        file_path : str
            Path to the csv file containing raw tick data.

        Notes
        -----
        This method reads the first row of the file and then uses `_assert_csv` to validate its format.
        """
        # Read in the first row & assert format
        first_row = pd.read_csv(file_path, nrows=1)
        self._assert_csv(first_row)

    def run(self, raw_tick_data: Union[list, tuple, pd.DataFrame]) -> list:
        """
        Reads a List, Tuple, or Dataframe and then constructs the financial data structure in the form of a list.
        The List, Tuple, or DataFrame must have only 3 columns: date_time, price, & volume.

        Parameters
        ----------
        raw_tick_data : list, tuple, or pd.DataFrame
            Raw tick data in the format [date_time, price, volume]

        Returns
        -------
        list
            Financial data structure
        """

        if isinstance(raw_tick_data, (list, tuple)):
            values = raw_tick_data

        elif isinstance(raw_tick_data, pd.DataFrame):
            # Check if the DataFrame has 3 columns, if yes, return Dataframe.values, if not reset index
            if len(raw_tick_data.columns) == 3:
                values = raw_tick_data.values

            else:
                # Reset index to move DateTimeIndex to a column
                idx_reset = raw_tick_data.reset_index()
                values = idx_reset.values

        else:
            raise ValueError(
                "The `raw_tick_data` is neither list nor tuple nor pd.DataFrame"
            )

        list_bars = self._extract_bars(raw_tick_data=values)

        # Set flag to True: notify function to use cache
        self.flag = True

        return list_bars

    @abstractmethod
    def _extract_bars(self, raw_tick_data: pd.DataFrame) -> list:
        """
        This method is required by all the bar types and is used to create the desired bars.

        Parameters
        ----------
        raw_tick_data : pd.DataFrame
            Contains 3 columns - 'date_time', 'price', and 'volume'.

        Returns
        -------
        list
            Bars built using the current batch.
        """

    @abstractmethod
    def _reset_cache(self):
        """
        This method is required by all the bar types. It describes how cache should be reset when new bar is sampled.
        """

    @staticmethod
    def _assert_csv(test_batch: pd.DataFrame):
        """
        Tests that the csv file read has the format: date_time, price, and volume.
        If not then the user needs to create such a file. This format is in place to remove any unwanted overhead.

        Parameters
        ----------
        test_batch : pd.DataFrame
            The first row of the dataset.

        Raises
        ------
        ValueError
            If the csv file format is invalid.
        """
        assert (
            test_batch.shape[1] == 3
        ), "CSV file must have exactly 3 columns: `date_time`, `price`, and `volume`."
        assert isinstance(
            test_batch.iloc[0, 1], float
        ), "Price column in CSV file must be of type float."
        assert not isinstance(
            test_batch.iloc[0, 2], str
        ), "Volume column in CSV file must be of type int or float."

        try:
            pd.to_datetime(test_batch.iloc[0, 0])
        except ValueError:
            raise ValueError(
                "Date-time column in CSV file is not in a valid format:",
                test_batch.iloc[0, 0],
            )

    def _update_high_low(self, price: float) -> Union[float, float]:
        """
        Update the high and low prices using the current price.

        Parameters
        ----------
        price : float
            The current price.

        Returns
        -------
        Tuple[float, float]
            A tuple containing the updated high and low prices.
        """
        if price > self.high_price:
            high_price = price
        else:
            high_price = self.high_price

        if price < self.low_price:
            low_price = price
        else:
            low_price = self.low_price

        return high_price, low_price

    def _create_bars(
        self,
        date_time: str,
        price: float,
        high_price: float,
        low_price: float,
        list_bars: list,
    ) -> None:
        """
        Given the inputs, construct a bar which has the following fields: date_time, open, high, low, close, volume, cum_buy_volume, cum_ticks, cum_dollar_value.

        Parameters
        ----------
        date_time : str
            Timestamp of the bar
        price : float
            The current price
        high_price : float
            Highest price in the period
        low_price : float
            Lowest price in the period
        list_bars : list
            List to which we append the bars

        Notes
        -----
        The date_time, price, and volume are expected to be consistent with the input data format specified in batch_run.
        """
        # Create bars
        open_price = self.open_price
        high_price = max(high_price, open_price)
        low_price = min(low_price, open_price)
        close_price = price
        volume = self.cum_statistics["cum_volume"]
        cum_buy_volume = self.cum_statistics["cum_buy_volume"]
        cum_ticks = self.cum_statistics["cum_ticks"]
        cum_dollar_value = self.cum_statistics["cum_dollar_value"]

        # Update bars
        list_bars.append(
            [
                date_time,
                self.tick_num,
                open_price,
                high_price,
                low_price,
                close_price,
                volume,
                cum_buy_volume,
                cum_ticks,
                cum_dollar_value,
            ]
        )

    def _apply_tick_rule(self, price: float) -> int:
        """
        This method updates the previous price used for tick rule calculations.

        Parameters
        ----------
        price : float
            Price at time t

        Returns
        -------
        int
            The signed tick

        Notes
        -----
        Applies the tick rule as defined on page 29 of Advances in Financial Machine Learning.
        """
        if self.prev_price is not None:
            tick_diff = price - self.prev_price
        else:
            tick_diff = 0

        if tick_diff != 0:
            signed_tick = np.sign(tick_diff)
            self.prev_tick_rule = signed_tick
        else:
            signed_tick = self.prev_tick_rule

        self.prev_price = price  # Update previous price used for tick rule calculations
        return signed_tick

    def _get_imbalance(self, price: float, signed_tick: int, volume: float) -> float:
        """
        Get the imbalance at a point in time, denoted as Theta_t.

        Parameters
        ----------
        price : float
            Price at t
        signed_tick : int
            signed tick, using the tick rule
        volume : float
            Volume traded at t

        Returns
        -------
        float
            Imbalance at time t

        Notes
        -----
        Advances in Financial Machine Learning, page 29.
        """
        if (
            self.inform_bar_type == "tick_imbalance"
            or self.inform_bar_type == "tick_run"
        ):
            imbalance = signed_tick
        elif (
            self.inform_bar_type == "dollar_imbalance"
            or self.inform_bar_type == "dollar_run"
        ):
            imbalance = signed_tick * volume * price
        elif (
            self.inform_bar_type == "volume_imbalance"
            or self.inform_bar_type == "volume_run"
        ):
            imbalance = signed_tick * volume
        else:
            raise ValueError(
                "Unknown information bar type, possible values are tick/dollar/volume imbalance/run type."
            )
        return imbalance


class BaseImbalanceBars(BaseBars):
    """
    Base class for Imbalance Bars (EMA and Const) which implements imbalance bars calculation logic.
    """

    def __init__(
        self,
        immb_bar_type: str,
        batch_size: int,
        expected_imbalance_window: int,
        exp_num_ticks_init: int,
        analyse_thresholds: bool,
    ):
        """
        Constructor

        Parameters
        ----------
        immb_bar_type : str
            Type of imbalance bar to create. Options: `tick_imbalance`, `dollar_imbalance`, `volume_imbalance`.
        batch_size : int, optional
            Number of rows to read in from the csv file (default is 2e7).
        expected_imbalance_window : int
            Window used to estimate expected imbalance from previous trades.
        exp_num_ticks_init : int
            Initial estimate for expected number of ticks in bar.
        analyse_thresholds : bool
            Flag to return thresholds values (theta, exp_num_ticks, exp_imbalance) in a form of Pandas DataFrame.
        """
        BaseBars.__init__(self, immb_bar_type, batch_size)

        self.expected_imbalance_window = expected_imbalance_window

        self.thresholds = {
            "cum_theta": 0,
            "expected_imbalance": np.nan,
            "exp_num_ticks": exp_num_ticks_init,
        }

        # Previous bars number of ticks and previous tick imbalances
        self.imbalance_tick_statistics = {"num_ticks_bar": [], "imbalance_array": []}

        if analyse_thresholds is True:
            # Array of dicts: {'timestamp': value, 'cum_theta': value, 'exp_num_ticks': value, 'exp_imbalance': value}
            self.bars_thresholds = []
        else:
            self.bars_thresholds = None

    def _reset_cache(self):
        """
        Implementation of abstract method `_reset_cache` for imbalance bars.
        """
        self.open_price = None
        self.high_price, self.low_price = -np.inf, np.inf
        self.cum_statistics = {
            "cum_ticks": 0,
            "cum_dollar_value": 0,
            "cum_volume": 0,
            "cum_buy_volume": 0,
        }
        self.thresholds["cum_theta"] = 0

    def _extract_bars(self, raw_tick_data: Tuple[dict, pd.DataFrame]) -> list:
        """
        For loop which compiles the various imbalance bars: dollar, volume, or tick.

        Parameters
        ----------
        raw_tick_data : pd.DataFrame or dict.
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

            # Bar statistics calculations
            self.cum_statistics["cum_ticks"] += 1
            self.cum_statistics["cum_dollar_value"] += dollar_value
            self.cum_statistics["cum_volume"] += volume
            if signed_tick == 1:
                self.cum_statistics["cum_buy_volume"] += volume

            # Imbalance calculations
            imbalance = self._get_imbalance(price, signed_tick, volume)
            self.imbalance_tick_statistics["imbalance_array"].append(imbalance)
            self.thresholds["cum_theta"] += imbalance

            # Get expected imbalance for the first time, when num_ticks_init passed
            if not list_bars and np.isnan(self.thresholds["expected_imbalance"]):
                self.thresholds["expected_imbalance"] = self._get_expected_imbalance(
                    self.expected_imbalance_window
                )

            if self.bars_thresholds is not None:
                self.thresholds["timestamp"] = date_time
                self.bars_thresholds.append(dict(self.thresholds))

            # Check expression for possible bar generation
            if (
                np.abs(self.thresholds["cum_theta"])
                > self.thresholds["exp_num_ticks"]
                * np.abs(self.thresholds["expected_imbalance"])
                if ~np.isnan(self.thresholds["expected_imbalance"])
                else False
            ):
                self._create_bars(
                    date_time, price, self.high_price, self.low_price, list_bars
                )

                self.imbalance_tick_statistics["num_ticks_bar"].append(
                    self.cum_statistics["cum_ticks"]
                )
                # Expected number of ticks based on formed bars
                self.thresholds["exp_num_ticks"] = self._get_exp_num_ticks()
                # Get expected imbalance
                self.thresholds["expected_imbalance"] = self._get_expected_imbalance(
                    self.expected_imbalance_window
                )
                # Reset counters
                self._reset_cache()

        return list_bars

    def _get_expected_imbalance(self, window: int):
        """
        Calculate the expected imbalance: 2P[b_t=1]-1, using an exponentially weighted moving average (EWMA), as described in the batch run documentation (pg 29).

        Parameters
        ----------
        window : int
            EWMA window for calculation

        Returns
        -------
        expected_imbalance : np.ndarray
            2P[b_t=1]-1, approximated using a EWMA.

        Notes
        -----
        Unconditional probability that a tick formulas in page 29.
        """
        if (
            len(self.imbalance_tick_statistics["imbalance_array"])
            < self.thresholds["exp_num_ticks"]
        ):
            # Waiting for array to fill for ewma
            ewma_window = np.nan
        else:
            # ewma window can be either the window specified in a function call
            # or it is len of imbalance_array if window > len(imbalance_array)
            ewma_window = int(
                min(len(self.imbalance_tick_statistics["imbalance_array"]), window)
            )

        if np.isnan(ewma_window):
            # return nan, wait until len(self.imbalance_array) >= self.exp_num_ticks_init
            expected_imbalance = np.nan
        else:
            expected_imbalance = ewma(
                np.array(
                    self.imbalance_tick_statistics["imbalance_array"][-ewma_window:],
                    dtype=float,
                ),
                window=ewma_window,
            )[-1]

        return expected_imbalance

    @abstractmethod
    def _get_exp_num_ticks(self):
        """
        Abstract method which updates expected number of ticks when new run bar is formed.
        """


# pylint: disable=too-many-instance-attributes
class BaseRunBars(BaseBars):
    """
    Base class for Run Bars (EMA and Const) which implements run bars calculation logic.
    """

    def __init__(
        self,
        run_bar_type: str,
        batch_size: int,
        num_prev_bars: int,
        expected_imbalance_window: int,
        exp_num_ticks_init: int,
        analyse_thresholds: bool,
    ):
        """
        Constructor

        Parameters
        ----------
        run_bar_type : str
            Type of imbalance bar to create. Options: `tick_run`, `dollar_run`, `volume_run`.
        batch_size : int, optional
            Number of rows to read in from the csv file (default is 2e7).
        num_prev_bars : int
            Number of previous bars to consider when calculating the expected number of ticks.
        expected_imbalance_window : int
            Window used to estimate expected imbalance from previous trades.
        exp_num_ticks_init : int
            Initial estimate for expected number of ticks in bar
        analyse_thresholds : bool
            Flag to return thresholds values (theta, exp_num_ticks, exp_runs) in a form of Pandas DataFrame.
        """
        BaseBars.__init__(self, run_bar_type, batch_size)

        self.num_prev_bars = num_prev_bars
        self.expected_imbalance_window = expected_imbalance_window

        self.thresholds = {
            "cum_theta_buy": 0,
            "cum_theta_sell": 0,
            "exp_imbalance_buy": np.nan,
            "exp_imbalance_sell": np.nan,
            "exp_num_ticks": exp_num_ticks_init,
            "exp_buy_ticks_proportion": np.nan,
            "buy_ticks_num": 0,
        }

        # Previous bars number of ticks and previous tick imbalances
        self.imbalance_tick_statistics = {
            "num_ticks_bar": [],
            "imbalance_array_buy": [],
            "imbalance_array_sell": [],
            "buy_ticks_proportion": [],
        }

        if analyse_thresholds:
            # Array of dicts: {'timestamp': value, 'cum_theta': value, 'exp_num_ticks': value, 'exp_imbalance': value}
            self.bars_thresholds = []
        else:
            self.bars_thresholds = None

        self.warm_up_flag = False

    def _reset_cache(self):
        """
        Implementation of abstract method `_reset_cache` for run bars.
        """
        self.open_price = None
        self.high_price, self.low_price = -np.inf, np.inf
        self.cum_statistics = {
            "cum_ticks": 0,
            "cum_dollar_value": 0,
            "cum_volume": 0,
            "cum_buy_volume": 0,
        }
        (
            self.thresholds["cum_theta_buy"],
            self.thresholds["cum_theta_sell"],
            self.thresholds["buy_ticks_num"],
        ) = (0, 0, 0)

    def _extract_bars(self, raw_tick_data: Tuple[list, np.ndarray]) -> list:
        """
        For loop which compiles the various run bars: dollar, volume, or tick.

        Parameters
        ----------
        raw_tick_data : list or np.ndarray
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

            # Bar statistics calculations
            self.cum_statistics["cum_ticks"] += 1
            self.cum_statistics["cum_dollar_value"] += dollar_value
            self.cum_statistics["cum_volume"] += volume
            if signed_tick == 1:
                self.cum_statistics["cum_buy_volume"] += volume

            # Imbalance calculations
            imbalance = self._get_imbalance(price, signed_tick, volume)

            if imbalance > 0:
                self.imbalance_tick_statistics["imbalance_array_buy"].append(imbalance)
                self.thresholds["cum_theta_buy"] += imbalance
                self.thresholds["buy_ticks_num"] += 1
            elif imbalance < 0:
                self.imbalance_tick_statistics["imbalance_array_sell"].append(
                    abs(imbalance)
                )
                self.thresholds["cum_theta_sell"] += abs(imbalance)

            self.warm_up_flag = np.isnan(
                [
                    self.thresholds["exp_imbalance_buy"],
                    self.thresholds["exp_imbalance_sell"],
                ]
            ).any()  # Flag indicating that one of imbalances is not counted (warm-up)

            # Get expected imbalance for the first time, when num_ticks_init passed
            if not list_bars and self.warm_up_flag:
                self.thresholds["exp_imbalance_buy"] = self._get_expected_imbalance(
                    self.imbalance_tick_statistics["imbalance_array_buy"],
                    self.expected_imbalance_window,
                    warm_up=True,
                )
                self.thresholds["exp_imbalance_sell"] = self._get_expected_imbalance(
                    self.imbalance_tick_statistics["imbalance_array_sell"],
                    self.expected_imbalance_window,
                    warm_up=True,
                )

                if (
                    bool(
                        np.isnan(
                            [
                                self.thresholds["exp_imbalance_buy"],
                                self.thresholds["exp_imbalance_sell"],
                            ]
                        ).any()
                    )
                    is False
                ):
                    self.thresholds["exp_buy_ticks_proportion"] = (
                        self.thresholds["buy_ticks_num"]
                        / self.cum_statistics["cum_ticks"]
                    )

            if self.bars_thresholds is not None:
                self.thresholds["timestamp"] = date_time
                self.bars_thresholds.append(dict(self.thresholds))

            # Check expression for possible bar generation
            max_proportion = max(
                self.thresholds["exp_imbalance_buy"]
                * self.thresholds["exp_buy_ticks_proportion"],
                self.thresholds["exp_imbalance_sell"]
                * (1 - self.thresholds["exp_buy_ticks_proportion"]),
            )

            # Check expression for possible bar generation
            max_theta = max(
                self.thresholds["cum_theta_buy"], self.thresholds["cum_theta_sell"]
            )
            if max_theta > self.thresholds[
                "exp_num_ticks"
            ] * max_proportion and not np.isnan(max_proportion):
                self._create_bars(
                    date_time, price, self.high_price, self.low_price, list_bars
                )

                self.imbalance_tick_statistics["num_ticks_bar"].append(
                    self.cum_statistics["cum_ticks"]
                )
                self.imbalance_tick_statistics["buy_ticks_proportion"].append(
                    self.thresholds["buy_ticks_num"] / self.cum_statistics["cum_ticks"]
                )

                # Expected number of ticks based on formed bars
                self.thresholds["exp_num_ticks"] = self._get_exp_num_ticks()

                # Expected buy ticks proportion based on formed bars
                exp_buy_ticks_proportion = ewma(
                    np.array(
                        self.imbalance_tick_statistics["buy_ticks_proportion"][
                            -self.num_prev_bars :
                        ],
                        dtype=float,
                    ),
                    self.num_prev_bars,
                )[-1]
                self.thresholds["exp_buy_ticks_proportion"] = exp_buy_ticks_proportion

                # Get expected imbalance
                self.thresholds["exp_imbalance_buy"] = self._get_expected_imbalance(
                    self.imbalance_tick_statistics["imbalance_array_buy"],
                    self.expected_imbalance_window,
                )
                self.thresholds["exp_imbalance_sell"] = self._get_expected_imbalance(
                    self.imbalance_tick_statistics["imbalance_array_sell"],
                    self.expected_imbalance_window,
                )

                # Reset counters
                self._reset_cache()

        return list_bars

    def _get_expected_imbalance(self, array: list, window: int, warm_up: bool = False):
        """
        Calculate the expected imbalance: 2P[b_t=1]-1, using an exponentially weighted moving average (EWMA), as described in the batch run documentation (pg 29).

        Parameters
        ----------
        array : list
            List of imbalances.
        window : int
            EWMA window for calculation.
        warm_up : bool
            Flag of whether warm up period passed.

        Returns
        -------
        expected_imbalance : np.ndarray
            2P[b_t=1]-1, approximated using a EWMA.

        Notes
        -----
        Unconditional probability that a tick formulas in page 29.
        """
        if len(array) < self.thresholds["exp_num_ticks"] and warm_up is True:
            # Waiting for array to fill for ewma
            ewma_window = np.nan
        else:
            # ewma window can be either the window specified in a function call
            # or it is len of imbalance_array if window > len(imbalance_array)
            ewma_window = int(min(len(array), window))

        if np.isnan(ewma_window):
            # return nan, wait until len(self.imbalance_array) >= self.exp_num_ticks_init
            expected_imbalance = np.nan
        else:
            expected_imbalance = ewma(
                np.array(array[-ewma_window:], dtype=float), window=ewma_window
            )[-1]

        return expected_imbalance

    @abstractmethod
    def _get_exp_num_ticks(self):
        """
        Abstract method which updates expected number of ticks when new imbalance bar is formed.
        """
