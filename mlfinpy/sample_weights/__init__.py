"""
Contains the code for implementing sample weights.
"""

from mlfinpy.sample_weights.attribution import (
    _apply_weight_by_return,
    get_weights_by_return,
    get_weights_by_time_decay,
)

__all__ = ["get_weights_by_time_decay", "get_weights_by_return", "_apply_weight_by_return"]
