import pandas as pd
import numpy as np
import pytest

from mushu.gen_eda.check_nans import check_nan_fill_values, check_nans


@pytest.mark.parametrize(
    "df, columns, minimum_threshold, expected_result",
    [
        [
            pd.DataFrame({"a": [1, 1, 2, 2], "b": [1, 2, 3, 4]}),
            "a",
            0.5,
            None,
        ],
        [
            pd.DataFrame({"a": [-1.1, -9.4, -5.3, 99, -0.9, 99]}),
            "a",
            0.5,
            (99, 2),
        ],
        [
            pd.DataFrame({"a": [1.1, 2, 3.7, 5.8, -1, 9.2, -1]}),
            "a",
            0.5,
            (-1.0, 2),
        ],
        [
            pd.DataFrame({"a": [1.1, 2, 3.7, 5.8, -0.1, 9.2] + [-99] * 10}),
            "a",
            0.5,
            (-99, 10),
        ],
    ],
    ids=[
        "No nan-likes",
        "99 as nan-like placeholder",
        "-1 as nan-like placeholder",
        "multiple positives/negatives, but -99 as a nan-like placeholder",
    ],
)
def test_check_nan_likes(df, columns, minimum_threshold, expected_result):
    """
    Tests that the function check_nan_likes works as expected.
    """

    # Assert
    assert check_nan_fill_values(df, columns, minimum_threshold) == expected_result


@pytest.mark.parametrize(
    "df, columns, minimum_threshold, expected_result",
    [
        [
            pd.DataFrame({"a": [1, 1, 1, 1], "b": [1, 2, 3, 4]}),
            "a",
            0.3,
            {"a": None},
        ],
        [
            pd.DataFrame({"a": [1, 1, 1, 1], "b": [1, 2, 3, 4]}),
            ["a", "b"],
            0.3,
            {"a": None, "b": None},
        ],
        [
            pd.DataFrame({"a": [1, None, 1, None], "b": [1, 2, 3, 4]}),
            "a",
            0.3,
            {"a": (None, 2)},
        ],
        [
            pd.DataFrame({"a": [1, None, 1, None], "b": [np.nan, 2, 3, 4]}),
            ["a", "b"],
            0.3,
            {"a": (None, 2), "b": (None, 1)},
        ],
        [
            pd.DataFrame({"a": [1, np.nan, 1, np.nan], "b": [np.nan, 2, 3, 4]}),
            ["a", "b"],
            0.3,
            {"a": (None, 2), "b": (None, 1)},
        ],
    ],
    ids=[
        "No nans, single column",
        "No nans, multiple columns",
        "nans, single column",
        "nans, multiple columns",
        "nans with np.nan, multiple columns",
    ],
)
def test_check_nans_with_nans(df, columns, minimum_threshold, expected_result):
    """
    Tests that the function check_nans works as expected.
    """

    # Assert
    assert check_nans(df, columns, minimum_threshold) == expected_result


@pytest.mark.parametrize(
    "df, columns, minimum_threshold, expected_result",
    [
        [
            pd.DataFrame({"a": [-1, 1.1, -2, 3.8, -10, 4.7], "b": [1, 2, 3, 4, 5, 6]}),
            "a",
            0.3,
            {"a": None},
        ],
        [
            pd.DataFrame(
                {"a": [-1, 1.1, -2, 3.8, -10, 4.7], "b": [1, 2, -3, 4, -5, -6]}
            ),
            ["a", "b"],
            0.3,
            {"a": None, "b": None},
        ],
        [
            pd.DataFrame({"a": [-1, 1.1, 2, 3.8, -1, 4.7], "b": [1, 2, 3, 4, 5, 6]}),
            "a",
            0.3,
            {"a": (-1.0, 2)},
        ],
        [
            pd.DataFrame(
                {"a": [-1, 1.1, -1, 3.8, 10, 4.7], "b": [1.7, 2.4, 3.9, -2, 5.6, -2]}
            ),
            ["a", "b"],
            0.3,
            {"a": (-1.0, 2), "b": (-2, 2)},
        ],
    ],
    ids=[
        "No nan-likes, single column",
        "No nan-likes, multiple columns",
        "-1 as nan-like placeholder, single column",
        "-1 and -2 as nan-like placeholders for respective columns, multiple columns",
    ],
)
def test_check_nans_with_nan_like(df, columns, minimum_threshold, expected_result):
    """
    Tests that the function check_nans works as expected.
    """

    # Assert
    assert check_nans(df, columns, minimum_threshold) == expected_result
