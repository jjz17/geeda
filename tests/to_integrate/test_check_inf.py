import pandas as pd
import numpy as np
import pytest

from mushu.gen_eda.check_inf import check_inf


@pytest.mark.parametrize(
    "df, columns, expected_result",
    [
        [
            pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [-4.7, -0.6, 2.5, 3.9, 5.5]}),
            "a",
            {"a": 0},
        ],
        [
            pd.DataFrame(
                {"a": [1, np.inf, 3, np.inf, 5], "b": [-4.7, -0.6, 2.5, 3.9, 5.5]}
            ),
            "a",
            {"a": 2},
        ],
        [
            pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [-4.7, -0.6, 2.5, 3.9, 5.5]}),
            ["a", "b"],
            {"a": 0, "b": 0},
        ],
        [
            pd.DataFrame(
                {
                    "a": [1, np.inf, 3, np.inf, 5],
                    "b": [np.inf, np.inf, np.inf, np.inf, -0.6],
                }
            ),
            ["a", "b"],
            {"a": 2, "b": 4},
        ],
    ],
    ids=[
        "No inf, single column",
        "2 inf, single column",
        "No inf, multiple columns",
        "Multiple inf, multiple columns",
    ],
)
def test_check_inf(df, columns, expected_result):
    """
    Tests that the function check_inf works as expected.
    """

    # Act, Assert
    assert check_inf(df, columns) == expected_result
