import pandas as pd
from typing import List, Union, Callable

from utils import make_list, validate_columns


class Geeda:
    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df

    def apply(
        self,
        columns: Union[str, List[str]],
        column_functions: Union[Callable, List[Callable]],
        df_functions: Union[Callable, List[Callable]],
    ) -> None:
        validate_columns(df=self.df, columns=columns)
        columns = make_list(columns)
        functions = make_list(functions)

        for function in column_functions:
            for column in columns:
                function(self.df[column])

        for function in df_functions:
            function(self.df)
