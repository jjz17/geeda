import pandas as pd
import numpy as np
import pytest

from mushu.gen_eda.print_special_values_report import print_special_values_report


@pytest.mark.parametrize(
    "df, columns, expected_printout",
    [
        [
            pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [-4.7, -0.6, 2.5, 3.9, 5.5]}),
            "a",
            "Column a:\n\tNaN: None\n\tInf: 0\n",
        ],
        [
            pd.DataFrame(
                {"a": [1, np.nan, np.nan, 4, 5], "b": [-4.7, -0.6, 2.5, 3.9, 5.5]}
            ),
            "a",
            "Column a:\n\tNaN: (None, 2)\n\tInf: 0\n",
        ],
        [
            pd.DataFrame({"a": [np.inf, 2, 3, 4, 5], "b": [-4.7, -0.6, 2.5, 3.9, 5.5]}),
            "a",
            "Column a:\n\tNaN: None\n\tInf: 1\n",
        ],
        [
            pd.DataFrame(
                {"a": [np.nan, np.inf, np.nan, 4, 5], "b": [-4.7, -0.6, 2.5, 3.9, 5.5]}
            ),
            "a",
            "Column a:\n\tNaN: (None, 2)\n\tInf: 1\n",
        ],
        [
            pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [-4.7, -0.6, 2.5, 3.9, 5.5]}),
            ["a", "b"],
            "Column a:\n\tNaN: None\n\tInf: 0\nColumn b:\n\tNaN: None\n\tInf: 0\n",
        ],
        [
            pd.DataFrame(
                {"a": [1, np.nan, np.nan, 4, 5], "b": [-4.7, -0.6, np.nan, 3.9, 5.5]}
            ),
            ["a", "b"],
            "Column a:\n\tNaN: (None, 2)\n\tInf: 0\nColumn b:\n\tNaN: (None, 1)\n\tInf: 0\n",
        ],
        [
            pd.DataFrame(
                {"a": [np.inf, 2, 3, 4, 5], "b": [-4.7, -0.6, 2.5, np.inf, 5.5]}
            ),
            ["a", "b"],
            "Column a:\n\tNaN: None\n\tInf: 1\nColumn b:\n\tNaN: None\n\tInf: 1\n",
        ],
        [
            pd.DataFrame(
                {
                    "a": [np.nan, np.inf, np.nan, 4, 5],
                    "b": [np.nan, np.inf, 2.5, 3.9, 5.5],
                }
            ),
            ["a", "b"],
            "Column a:\n\tNaN: (None, 2)\n\tInf: 1\nColumn b:\n\tNaN: (None, 1)\n\tInf: 1\n",
        ],
    ],
    ids=[
        "No special values, single column",
        "No inf, single column",
        "No nan, single column",
        "Nan and inf, single column",
        "No special values, multiple columns",
        "No inf, multiple columns",
        "No nan, multiple columns",
        "Nan and inf, multiple columns",
    ],
)
def test_print_special_values_report(df, columns, expected_printout, capsys):
    """
    Tests that the function special_values_report works as expected.
    """

    # Act
    print_special_values_report(df, columns)

    # Assert
    assert expected_printout in capsys.readouterr().out
