"""
Various helper functions
"""

from typing import List

import numpy as np
import pandas as pd


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
