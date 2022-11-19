from enum import Enum
from functools import partial
import pandas as pd
from typing import List, Union, Optional
from mushu.gen_eda.check_nans import check_nans
from mushu.gen_eda.check_inf import check_inf
from mushu.gen_eda.utils import get_list, validate_dataframe_columns


class SpecialValueReports(Enum):
    NaN = partial(check_nans)
    Inf = partial(check_inf)


def print_special_values_report(
    df: pd.DataFrame,
    columns: Union[str, List[str]] = None,
    reports: Optional[List[SpecialValueReports]] = None,
) -> None:
    """
    Generate printout report of NaN and Inf values in given columns.

    :param df:
        input DataFrame
    :param columns:
        target column(s) to analyze, default is all columns
    :param reports:
        list of special values to report on, default is all defined special values
    """
    if reports is None:
        reports = [r for r in SpecialValueReports]

    columns = df.columns if columns is None else get_list(columns)

    validate_dataframe_columns(df, columns)

    generated_reports = {
        report.name: SpecialValueReports[report.name].value(df=df, columns=columns)
        for report in reports
    }

    for column in columns:
        column_printout = f"Column {column}:"
        for value, generated_report in generated_reports.items():
            column_printout += f"\n\t{value}: {generated_report[column]}"
        print(column_printout)
