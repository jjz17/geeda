from calendar import c
import pandas as pd


def is_categorical(column: pd.Series, min_threshold: float = 0.3, dropna: bool = True) -> bool:
    """
    Identify if the given column is of a categorical data type.

    Args:
        column (pd.Series): 
            The column to analyze
        min_threshold (float, optional): 
            The minimum threshold for the ratio between unique values and total values
            for the column to be considered categorical, defaults to 0.3.
        dropna (bool, optional): 
            Drops na values before analyzing if True, defaults to True.

    Returns:
        bool: 
            True if the column is categorical
    """

    if dropna:
        column = column.dropna()

    unique_count = column.nunique()
    return unique_count/column.size >= min_threshold
