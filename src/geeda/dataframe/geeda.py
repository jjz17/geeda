import pandas as pd
from typing import List, Union, Callable, Optional
import inspect

from src.geeda.utils import make_list, validate_columns


class Geeda:
    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df

    def apply(
        self,
        eda_functions: Union[Callable, List[Callable]],
        columns: Optional[Union[str, List[str]]] = None,
    ) -> None:
        """
        Apply the given EDA functions on the given columns of the dataframe.

        Args:
            eda_functions (Union[Callable, List[Callable]]):
                The EDA function(s) to apply on the columns
            columns (Optional[Union[str, List[str]]], optional):
                The column(s) to apply the EDA functions on, defaults to all columns in the dataframe
        """
        columns = (
            self.df.columns
            if columns is None
            else validate_columns(df=self.df, columns=columns)
        )
        eda_funcs: List[Callable] = make_list(eda_functions)

        # Separate column and dataframe functions
        column_functions = []
        df_functions = []
        for function in eda_funcs:
            import_path: str = inspect.getmodule(function).__name__
            if "column" in import_path:
                column_functions.append(function)
            else:
                df_functions.append(function)

        column_results = dict()
        column_printout_idx = [function.__name__ for function in column_functions]
        for column in columns:
            single_column_results = []
            for function in column_functions:
                result = function(self.df[column])
                single_column_results.append(result)
            column_results[column] = single_column_results

        df_results = []
        df_printout_idx = [function.__name__ for function in df_functions]
        for function in df_functions:
            result = function(self.df)
            df_results.append(result)
