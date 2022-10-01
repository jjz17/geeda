import pytest
import pandas as pd

from src.geeda.column.is_categorical import is_categorical


@pytest.mark.parametrize(
    argnames="column, max_threshold, expected",
    argvalues=[
        [pd.Series([1, 1, 1, 1, 1, 2, 2, 2, 2, 2]), 0.3, True],
        [pd.Series([1, 1, 1, 2, 2, 3, 3, 3, 4, 4]), 0.5, True],
        [pd.Series([1, 1, 1, 2, 2, 3, 3, 3, 4, 4]), 0.3, False],
        [pd.Series([1, 2, 3, 4, 5]), 0.3, False],
    ],
    ids=[
        "Categorical with 2 unique values",
        "Categorical with 4 unique values",
        "Not categorical with 4 unique values",
        "Not categorical with all unique values",
    ],
)
def test_is_categorical(column, max_threshold, expected):
    # Act, Assert
    assert is_categorical(column, max_threshold) == expected
