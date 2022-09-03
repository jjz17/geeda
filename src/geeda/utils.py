import pandas as pd
from typing import List, Union


def make_list(input: any):
    if not isinstance(input, list):
        input = list(input)
    return input


def validate_columns(df: pd.DataFrame, columns: Union[str, List[str]]) -> None:
    columns = make_list(columns)
    df_columns = df.columns

    missing_columns = []
    for column in columns:
        if column not in df_columns:
            missing_columns.append(column)
    if len(missing_columns) > 0:
        raise KeyError(f"Column(s) {missing_columns} is/are not in the DataFrame")
