"""Synthetic regression outputs captured from licensed mlfinlab 2.3.0.

No legacy source, credentials or production data are included. The fixture
records black-box event positions, not the proprietary implementation.
"""

import json
import unittest
from pathlib import Path

import pandas as pd

from mlfinpy.filters.filters import z_score_filter
from mlfinpy.labeling.labeling import add_vertical_barrier


class TestParityPatches(unittest.TestCase):
    def setUp(self):
        self.index = pd.to_datetime(
            ["2024-01-01", "2024-01-02", "2024-01-04", "2024-01-05", "2024-01-08", "2024-01-09"],
            utc=True,
        )
        self.close = pd.Series(range(6), index=self.index)

    def test_count_barriers_override_duration_and_trim_tail(self):
        events = self.index[[0, 2, 4, 5]]
        for count, starts, ends in [(0, [0, 2, 4, 5], [0, 2, 4, 5]), (1, [0, 2, 4], [1, 3, 5]), (2, [0, 2], [2, 4])]:
            with self.subTest(count=count):
                actual = add_vertical_barrier(events, self.close, num_days=100, num_bars=count)
                expected = pd.Series(self.index[ends], index=self.index[starts])
                pd.testing.assert_series_equal(actual, expected)

    def test_empty_or_out_of_range_count_barriers(self):
        for events, count in [(self.index[:0], 1), (self.index, 6)]:
            with self.subTest(count=count, events=len(events)):
                actual = add_vertical_barrier(events, self.close, num_bars=count)
                expected = pd.Series(self.index[:0], index=self.index[:0])
                pd.testing.assert_series_equal(actual, expected)

    def test_count_barriers_require_events_in_close(self):
        events = pd.DatetimeIndex([pd.Timestamp("2024-01-01 12:00", tz="UTC")])
        with self.assertRaises(KeyError):
            add_vertical_barrier(events, self.close, num_bars=1)

    def test_filter_matches_licensed_golden_outputs(self):
        fixture = json.loads((Path(__file__).parent / "test_data" / "poseidon_parity_2308.json").read_text())
        for case in fixture["filters"]:
            with self.subTest(case=case):
                index = pd.date_range("2024-01-01", periods=len(fixture["series"][case["series"]]), freq="h", tz="UTC")
                series = pd.Series(fixture["series"][case["series"]], dtype=case["dtype"], index=index)
                expected = index[case["events"]]
                for timestamps in [True, False]:
                    actual = z_score_filter(
                        series,
                        case["mean_window"],
                        case["std_window"],
                        case["z_score"],
                        time_stamps=timestamps,
                        influence=case["influence"],
                    )
                    pd.testing.assert_index_equal(actual, expected)

    def test_filter_threshold_is_strict(self):
        series = pd.Series([1.0] * 6, index=pd.date_range("2024-01-01", periods=6, freq="h"))
        self.assertEqual(len(z_score_filter(series, 3, 3, influence=0.5)), 0)

    def test_short_filter_input_preserves_legacy_error(self):
        with self.assertRaises(IndexError):
            z_score_filter(pd.Series([1.0, 2.0]), 3, 3)
