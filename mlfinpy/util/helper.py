"""
Various helper functions
"""

import pandas as pd
import numpy as np
from typing import List


def crop_data_frame_in_batches(df: pd.DataFrame, chunksize: int) -> List[pd.DataFrame]:
    """
    Splits df into chunks of chunksize.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to split.
    chunksize : int
        Number of rows in chunk.

    Returns
    -------
    List[pd.DataFrame]
        Chunks (pd.DataFrames).
    """
    generator_object = []
    for _, chunk in df.groupby(np.arange(len(df)) // chunksize):
        generator_object.append(chunk)
    return generator_object
