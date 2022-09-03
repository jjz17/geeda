import pandas as pd
from typing import List, Union, Callable
import inspect

from utils import make_list, validate_columns


class Geeda:
    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df

    def apply(
        self,
        columns: Union[str, List[str]],
        eda_functions: Union[Callable, List[Callable]],
    ) -> None:
        columns = validate_columns(df=self.df, columns=columns)
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

        for function in column_functions:
            for column in columns:
                function(self.df[column])

        for function in df_functions:
            function(self.df)
