from calendar import c
from typing import List, Dict, Any


class Printout:
    """
    Wrapper for dict of check/metric mapped to results?
    """

    def __init__(
        self,
        columns: List[str],
        column_results: Dict[str, List[Any]],
        df_results: Dict[str, Any],
    ) -> None:
        self.columns = columns
        self.column_results = column_results
        self.df_results = df_results

    def printout(self):
        pass
