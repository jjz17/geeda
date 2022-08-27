import pytest
import pandas as pd

from src.geeda.column.is_categorical import is_categorical


@pytest.mark.parametrize(
    argnames=["column", "max_threshold", "expected"],
    argvalues=[[pd.Series([1, 1, 1, 1, 1, 2, 2, 2, 2, 2]), 0.3, True]],
    ids=["1"],
)
def test_is_categorical(column, max_threshold, expected):
    assert is_categorical(column, max_threshold) == expected
