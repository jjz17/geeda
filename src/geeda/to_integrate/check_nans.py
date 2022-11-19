from typing import List, Union, Dict, Tuple, Optional
from numbers import Number

import pandas as pd
import numpy as np

from mushu.gen_eda.is_categorical import is_categorical
from mushu.gen_eda.is_pseudo_categorical import is_pseudo_categorical
from mushu.gen_eda.utils import get_list, validate_dataframe_columns


def check_nan_fill_values(
    df: pd.DataFrame, column: str, minimum_threshold: float = 0.3
) -> Union[Tuple[Union[Number, None], int], None]:
    """
    Check if a NaN fill-value is present and count how many occurrences there are. A fill-value is defined as either
    a single positive or negative value out of all unique values, or one value that comprises more than the
    `minimum_threshold` of the entire column.

    NOTE: For NaN fill-values, this function can only identify what *might* be a NaN fill-value
    and is intended to be used for generating suggestions rather than resolute decisions.

    :param df:
        input DataFrame
    :param column:
        target column to analyze
    :param minimum_threshold:
        minimum threshold for ratio of the count of a unique value to the total count of the data
        before it is considered to be a NaN fill-value
    :return:
        a tuple, the first element being the NaN fill-value and the second element being the count of its occurrences
    """
    data = df[column]
    # Drop inf values
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna()
    value_counts = data.value_counts()
    records = data.size
    unique_vals = data.unique()

    # Check if only 1 unique value is positive/zero/negative
    positive_vals = list(filter(lambda x: x > 0, unique_vals))
    negative_vals = list(filter(lambda x: x < 0, unique_vals))
    if len(positive_vals) == 1:
        count = value_counts.loc[positive_vals[0]]
        return positive_vals[0], count
    elif len(negative_vals) == 1:
        count = value_counts.loc[negative_vals[0]]
        return negative_vals[0], count
    else:
        potential_nans = dict()
        for value, count in value_counts.items():
            if (count / records) > minimum_threshold:
                potential_nans[count] = value

        if len(potential_nans) == 0:
            return None
        else:
            # Return value with the largest count
            max_count = max(potential_nans.keys())
            val = potential_nans[max_count]
            return val, max_count


def check_nans(
    df: pd.DataFrame,
    columns: Union[str, List[str]] = None,
    minimum_threshold: float = 0.3,
) -> Dict[str, Optional[Tuple[Union[Number, None], int]]]:
    """
    Identify if NaNs or NaN fill-values are present in given columns.

    NOTE: For NaN fill-values, this function can only identify what *might* be a NaN fill-value
    and is intended to be used for generating suggestions rather than resolute decisions.

    :param df:
        input DataFrame
    :param columns:
        target column(s) to analyze, default is all columns
    :param minimum_threshold:
        minimum threshold for ratio of the count of a unique value to the total count of the data
        before it is considered to be a NaN fill-value
    :return:
        a dict with column names as keys and tuples as values, the first element being the NaN fill-value
        and the second element being the count of its occurrences
    """
    nan_counts: Dict[str, Optional[Tuple[Union[Number, None], int]]] = dict()

    columns = df.columns if columns is None else get_list(columns)

    validate_dataframe_columns(df, columns)

    for column in columns:
        data = df[column]

        # If no NaN values
        if data.isna().sum() == 0:
            # If categorical or pseudo-categorical
            if is_categorical(df, column) or is_pseudo_categorical(df, column):
                nan_counts[column] = None
            else:
                nan_counts[column] = check_nan_fill_values(
                    df, column, minimum_threshold
                )
        else:
            nan_counts[column] = (None, data.isna().sum())

    return nan_counts
