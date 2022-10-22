import pytest
import pandas as pd
from scipy import stats

from src.geeda.column.check_distribution import is_significant, check_distribution


@pytest.mark.parametrize(
    argnames="p_value, alpha, expected",
    argvalues=[
        [0.1, 0.05, False],
        [0.02, 0.05, True],
        [0.03, 0.01, False],
        [0.01, 0.01, True],
    ],
    ids=[
        "Not significant at 0.05 alpha",
        "Categorical with 4 unique values",
        "Not categorical with 4 unique values",
        "Not categorical with all unique values",
    ],
)
def test_is_significant(p_value, alpha, expected):
    # Act, Assert
    assert is_significant(p_value, alpha) == expected


@pytest.mark.parametrize(
    argnames="column, alpha, expected",
    argvalues=[
        [pd.Series([1, 1, 1, 1, 1, 2, 2, 2, 2, 2]), 0.05, "not normal, uniform"],
        [pd.Series([1, 1, 1, 2, 2, 3, 3, 3, 4, 4]), 0.05, "normal, uniform"],
        [pd.Series([1, 1, 1, 2, 2, 3, 3, 3, 4, 4]), 0.01, "normal, uniform"],
        [pd.Series([1, 2, 3, 4, 5]), 0.01, "normal, uniform"],
        [
            pd.Series(stats.norm.rvs(loc=1, scale=1, size=100, random_state=1)),
            0.05,
            "normal, not uniform",
        ],
        [
            pd.Series(stats.norm.rvs(loc=5, scale=10, size=100, random_state=1)),
            0.01,
            "normal, not uniform",
        ],
        [
            pd.Series(stats.uniform.rvs(loc=0, scale=1, size=100, random_state=1)),
            0.05,
            "not normal, not uniform",
        ],
        [
            pd.Series(stats.uniform.rvs(loc=1, scale=10, size=100, random_state=1)),
            0.01,
            "not normal, not uniform",
        ],
    ],
    ids=[
        "Categorical with 2 unique values",
        "Categorical with 4 unique values",
        "Not categorical with 4 unique values",
        "Not categorical with all unique values",
        "Categorical with 2 unique values",
        "Categorical with 4 unique values",
        "Not categorical with 4 unique values",
        "Not categorical with all unique values",
    ],
)
def test_check_distribution(column, alpha, expected):
    # Act, Assert
    assert check_distribution(column, alpha) == expected
