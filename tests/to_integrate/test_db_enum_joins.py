import pytest
import pandas as pd

from mushu.gen_eda.db_enum_joins import (
    get_enum_to_value_mappings,
    get_enum_joins_query,
    db_enum_joins,
)


@pytest.mark.parametrize(
    "main_df, dataframes, expected",
    [
        [
            pd.DataFrame({"x_id": [1, 1, 2, 2, 3, 3], "value": [1, 2, 2, 3, 3, 3]}),
            {"X": pd.DataFrame({"x_id": [1, 2, 3], "x_name": ["x1", "x2", "x3"]})},
            {"x_id": ("X", "x_name")},
        ],
        [
            pd.DataFrame({"x_id": [1, 1, 2, 2, 3, 3], "y_id": [1, 2, 2, 3, 3, 3]}),
            {
                "X": pd.DataFrame({"x_id": [1, 2, 3], "x_name": ["x1", "x2", "x3"]}),
                "Y": pd.DataFrame({"y_id": [1, 2, 3], "y_value": ["y1", "y2", "y3"]}),
            },
            {"x_id": ("X", "x_name"), "y_id": ("Y", "y_value")},
        ],
    ],
    ids=[
        "Single enum mapping",
        "Multiple enum mappings",
    ],
)
def test_get_enum_to_value_mappings(main_df, dataframes, expected):
    # Act, Assert
    assert get_enum_to_value_mappings(main_df, **dataframes) == expected


def test_get_enum_to_value_mappings_bad_columns():
    # Arrange
    main_df = pd.DataFrame({"x_id": [1], "value": [1]})
    enum_dfs = {
        "x": pd.DataFrame({"x_id": [1], "x_name": ["x1"], "x_something_else": [123]}),
    }

    # Act, Assert
    with pytest.raises(
        ValueError,
        match="Table: x contains the enum `x_id` but should only have 2 columns",
    ):
        _ = get_enum_to_value_mappings(main_df, **enum_dfs)


@pytest.mark.parametrize(
    "main_table_name, main_table_columns, enum_to_value_mapping, expected",
    [
        [
            "Main",
            ["x_id", "val1"],
            {"x_id": ("X", "x_val"), "y_id": ("Y", "y_val"), "z_id": ("Z", "z_val")},
            "SELECT X.x_val, Main.val1 FROM Main JOIN X ON X.x_id = Main.x_id",
        ],
        [
            "Main",
            ["x_id", "val1", "y_id", "val2"],
            {"x_id": ("X", "x_val"), "y_id": ("Y", "y_val"), "z_id": ("Z", "z_val")},
            "SELECT X.x_val, Y.y_val, Main.val1, Main.val2 FROM Main JOIN X ON X.x_id = Main.x_id"
            " JOIN Y ON Y.y_id = Main.y_id",
        ],
    ],
    ids=["Single enum columns", "Multiple enum columns"],
)
def test_get_enum_joins_query(
    main_table_name, main_table_columns, enum_to_value_mapping, expected
):
    # Act, Assert
    assert (
        get_enum_joins_query(main_table_name, main_table_columns, enum_to_value_mapping)
        == expected
    )


@pytest.mark.parametrize(
    "main_df, main_df_name, printout, dataframes, expected_query, expected_printout",
    [
        [
            pd.DataFrame({"x_id": [1, 1, 2, 2, 3, 3], "value": [1, 2, 2, 3, 3, 3]}),
            "MAIN",
            True,
            {"X": pd.DataFrame({"x_id": [1, 2, 3], "x_name": ["x1", "x2", "x3"]})},
            "SELECT X.x_name, MAIN.value FROM MAIN JOIN X ON X.x_id = MAIN.x_id",
            "JOIN X ON X.x_id = MAIN.x_id",
        ],
        [
            pd.DataFrame({"x_id": [1, 1, 2, 2, 3, 3], "y_id": [1, 2, 2, 3, 3, 3]}),
            "MAIN",
            False,
            {
                "X": pd.DataFrame({"x_id": [1, 2, 3], "x_name": ["x1", "x2", "x3"]}),
                "Y": pd.DataFrame({"y_id": [1, 2, 3], "y_value": ["y1", "y2", "y3"]}),
            },
            "SELECT X.x_name, Y.y_value FROM MAIN JOIN X ON X.x_id = MAIN.x_id JOIN Y ON Y.y_id = MAIN.y_id",
            "",
        ],
        [
            pd.DataFrame({"x_id": [1, 1, 2, 2, 3, 3], "y_id": [1, 2, 2, 3, 3, 3]}),
            "MAIN",
            True,
            {
                "X": pd.DataFrame({"x_id": [1, 2, 3], "x_name": ["x1", "x2", "x3"]}),
                "Y": pd.DataFrame({"y_id": [1, 2, 3], "y_value": ["y1", "y2", "y3"]}),
                "Z": pd.DataFrame({"z_id": [1, 2, 3], "z_size": ["z1", "z2", "z3"]}),
            },
            "SELECT X.x_name, Y.y_value FROM MAIN JOIN X ON X.x_id = MAIN.x_id JOIN Y ON Y.y_id = MAIN.y_id",
            "JOIN Y ON Y.y_id = MAIN.y_id",
        ],
    ],
    ids=[
        "Replace 1 enum column",
        "Replace 2 enum columns, no printout",
        "Replace 2 enum columns, include irrelevant enum value dataframe",
    ],
)
def test_db_enum_joins(
    main_df,
    main_df_name,
    printout,
    dataframes,
    expected_query,
    expected_printout,
    capsys,
):
    """
    Tests that the function db_enum_joins works as expected.
    """
    # Act, Assert
    assert (
        db_enum_joins(main_df, main_df_name, printout, **dataframes) == expected_query
    )
    assert expected_printout in capsys.readouterr().out


@pytest.mark.parametrize(
    "main_df, main_df_name, printout, dataframes, expected_exception, expected_message",
    [
        [
            pd.DataFrame({"x_id": [1, 1, 2, 2, 3, 3], "y_id": [1, 2, 2, 3, 3, 3]}),
            "MAIN",
            True,
            {
                "X": pd.DataFrame(
                    {
                        "x_id": [1, 2, 3],
                        "x_name": ["x1", "x2", "x3"],
                        "x_size": [4, 5, 6],
                    }
                ),
                "Y": pd.DataFrame({"y_id": [1, 2, 3], "y_value": ["y1", "y2", "y3"]}),
            },
            ValueError,
            r".*Table: X should only have 2 columns.*",
        ],
    ],
    ids=[
        "Enum value dataframe with more than 2 columns",
    ],
)
def test_raise_expections(
    main_df, main_df_name, printout, dataframes, expected_exception, expected_message
):
    """
    Tests that the function db_enum_joins raises exceptions as expected.
    """
    with pytest.raises(expected_exception, match=expected_message):
        db_enum_joins(main_df, main_df_name, printout, **dataframes)
