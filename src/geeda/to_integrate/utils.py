import pandas as pd
from typing import List, Union, Any


def get_list(argument: Union[Any, List[Any]]) -> List[Any]:
    """
    Return the input wrapped in a list, if it is not already a list or if it is not None.

    :param argument:
        Any object
    :return:
        The input if it is a list, the input wrapped in a list if it is not a list.
    """
    if isinstance(argument, list):
        return argument
    return [argument]


def validate_dataframe_columns(
    df: pd.DataFrame, columns: Union[str, List[str]]
) -> None:
    """
    Validate all columns to ensure they are present in the given dataframe,
    raises KeyError if one or more columns are not present.

    :param df:
        input DataFrame
    :param columns:
        target column(s) to analyze, default is all columns in `df`
    """
    missing_columns = []
    for column in columns:
        # Check column exists
        if column not in df.columns:
            missing_columns.append(column)
    if len(missing_columns) != 0:
        raise KeyError(f"The column(s) {missing_columns} are not in the DataFrame.")


def validate_no_nulls(arguments: dict) -> None:
    """
    Validate input arguments to ensure they are not null, raises Error if any argument is null.

    NOTE: Intended use case is to check if local variables are all non null (e.g. Call: validate_no_nulls(locals()))

    :param arguments: dictionary of arguments to validate
    """
    null_arguments = []
    for key, value in arguments.items():
        if isinstance(value, pd.DataFrame):
            if value.empty:
                null_arguments.append(key)
        else:
            if value is None:
                null_arguments.append(key)
    if len(null_arguments) != 0:
        raise TypeError(
            f"The argument(s) {null_arguments} must not be null. DataFrames must not be empty."
        )
