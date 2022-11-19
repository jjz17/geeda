import pandas as pd
from typing import List, Optional

from mushu.gen_eda.utils import get_list, validate_dataframe_columns


def print_duplicates_info(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    rounded: bool = False,
) -> None:
    """
    Helper function to analyze the selected columns for duplicate rows under specified conditions.

    :param df: input DataFrame
    :param columns: target column(s) to analyze
    :param rounded: input True if the numerical columns of `df` have been rounded, default is False
    """
    if columns is None:
        columns = df.columns

    original_len = len(df)
    processed_len = len(df.drop_duplicates(subset=columns))
    if rounded:
        df_id = "Rounded dataframe:"
    else:
        df_id = "Original dataframe:"
    duplicates_info = f"{original_len - processed_len} duplicate rows ({(1 - processed_len / original_len) * 100}%)"
    printout = f"\t{df_id} {duplicates_info}"
    print(printout)


def check_duplicates(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    round_precision: Optional[int] = None,
) -> None:
    """
    Analyze the selected columns for duplicate rows under specified conditions. If columns are not given,
    all columns will be included when checking.

    :param df:
        input DataFrame
    :param columns:
        target column(s) to analyze, default is all columns in `df`
    :param round_precision:
        number of decimals of precision to round float-type columns to, default is to skip rounding
    """
    columns = df.columns if columns is None else get_list(columns)

    validate_dataframe_columns(df, columns)

    # Quantify duplicate rows in columns of interest
    print_duplicates_info(df, columns)
    if round_precision is not None:
        df_rounded = df.round(round_precision)
        print_duplicates_info(df_rounded, columns, True)

    # Quantify duplicate rows in columns of interest by testing all subsets of size n-1
    if len(columns) > 1:
        for column in columns:
            columns_subset = [col for col in columns if col != column]
            print(f"Excluding column: {column}")
            print_duplicates_info(df, columns_subset)
            if round_precision is not None:
                print_duplicates_info(df_rounded, columns_subset, rounded=True)
