"""
Utility functions used by other modules to provide functionality
such as: multiprocessing, vectorization, fractional differentiation,
etc.
This module contains snippets from various chapters in the book.
For detail, refer to the specific function in the module.
"""

from mlfinpy.util.fast_ewma import ewma
from mlfinpy.util.frac_diff import (
    frac_diff,
    frac_diff_ffd,
    get_weights,
    get_weights_ffd,
)
from mlfinpy.util.generate_dataset import get_classification_data
from mlfinpy.util.multiprocess import (
    expand_call,
    lin_parts,
    mp_pandas_obj,
    nested_parts,
    process_jobs,
    process_jobs_,
    report_progress,
)
from mlfinpy.util.volatility import (
    get_daily_vol,
    get_garman_class_vol,
    get_parksinson_vol,
    get_yang_zhang_vol,
)

__all__ = [
    "ewma",
    "mp_pandas_obj",
    "nested_parts",
    "lin_parts",
    "expand_call",
    "report_progress",
    "process_jobs",
    "process_jobs_",
    "get_daily_vol",
    "get_garman_class_vol",
    "get_yang_zhang_vol",
    "get_parksinson_vol",
    "get_weights",
    "frac_diff",
    "get_weights_ffd",
    "frac_diff_ffd",
    "get_classification_data",
]
