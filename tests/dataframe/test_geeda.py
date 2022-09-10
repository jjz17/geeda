import pytest
import pandas as pd

from src.geeda.dataframe.geeda import Geeda
from src.geeda.column.is_categorical import is_categorical


@pytest.mark.parametrize(
    argnames="df, columns, eda_functions, expected",
    argvalues=[
        [
            pd.DataFrame({"a": range(1, 6), "b": range(6, 11)}),
            "a",
            is_categorical,
            "True",
        ],
    ],
    ids=["Basic"],
)
def test_geeda(df, columns, eda_functions, expected, capsys):
    # Arrange
    geeda = Geeda(df)

    # Act
    geeda.apply(columns, eda_functions)

    # Assert
    assert expected in capsys.readouterr().out
