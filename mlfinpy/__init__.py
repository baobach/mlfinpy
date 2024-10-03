"""
Mlfinpy is a toolbox adapted from the book Advances in Financial Machine Learning.

This package contains functions in the book that help you implement the ideas and
code snippets without worrying about structure your code.

This package is for acamedmic purpose only. Not meant for live trading or active portfolio management.
"""

# read version from installed package
from importlib.metadata import version

__version__ = version("mlfinpy")

import mlfinpy.data_structure as data_structure
import mlfinpy.dataset as dataset
import mlfinpy.filters.filters as filters
import mlfinpy.labeling as labeling
import mlfinpy.util as util

__all__ = [
    "data_structure",
    "filters",
    "labeling",
    "util",
    "dataset",
]
