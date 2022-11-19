# COPYRIGHTS AND PERMISSIONS:
# Copyright 2022 MORSECORP, Inc. All rights reserved.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import pandas as pd
import pytest

from mushu.gen_eda.check_duplicates import print_duplicates_info, check_duplicates


@pytest.mark.parametrize(
    "df, columns, rounded, expected_printout",
    [
        [
            pd.DataFrame({"a": [1, 1, 1, 1], "b": [2, 2, 3, 3]}),
            "a",
            False,
            "Original dataframe: 3 duplicate rows (75.0%)",
        ],
        [
            pd.DataFrame({"a": [1, 1, 1, 1], "b": [2, 2, 3, 3]}),
            ["a", "b"],
            False,
            "Original dataframe: 2 duplicate rows (50.0%)",
        ],
        [
            pd.DataFrame({"a": [1, 1, 1, 1], "b": [2, 2, 3, 3]}),
            "a",
            True,
            "Rounded dataframe: 3 duplicate rows (75.0%)",
        ],
        [
            pd.DataFrame({"a": [1, 1, 1, 1], "b": [2, 2, 3, 3]}),
            ["a", "b"],
            True,
            "Rounded dataframe: 2 duplicate rows (50.0%)",
        ],
        [
            pd.DataFrame({"a": [1, 1]}),
            None,
            True,
            "Rounded dataframe: 1 duplicate rows (50.0%)",
        ],
    ],
    ids=[
        "Original df, single column",
        "Original df, multiple columns",
        "Rounded df, single column",
        "Rounded df, multiple columns",
        "Rounded df, None columns, single column",
    ],
)
def test_print_duplicates_info(df, columns, rounded, expected_printout, capsys):
    # Act
    print_duplicates_info(df, columns, rounded)

    # Assert
    assert expected_printout in capsys.readouterr().out


@pytest.mark.parametrize(
    "df, columns, round_precision, expected_printouts",
    [
        [
            pd.DataFrame({"a": [1, 1, 1, 1], "b": [2, 2, 3, 3]}),
            "a",
            3,
            [
                "Original dataframe: 3 duplicate rows (75.0%)",
                "Rounded dataframe: 3 duplicate rows (75.0%)",
            ],
        ],
        [
            pd.DataFrame({"a": [1, 1, 1, 1], "b": [2, 2, 3, 3]}),
            ["a", "b"],
            3,
            [
                "Excluding column: a",
                "\tOriginal dataframe: 2 duplicate rows (50.0%)",
                "\tRounded dataframe: 2 duplicate rows (50.0%)",
                "Excluding column: b",
                "\tOriginal dataframe: 3 duplicate rows (75.0%)",
                "\tRounded dataframe: 3 duplicate rows (75.0%)",
            ],
        ],
        [
            pd.DataFrame({"a": [1.11, 1.12, 1.13, 1.14], "b": [2, 2, 3, 3]}),
            "a",
            1,
            [
                "Original dataframe: 0 duplicate rows (0.0%)",
                "Rounded dataframe: 3 duplicate rows (75.0%)",
            ],
        ],
        [
            pd.DataFrame({"a": [1.11, 1.12, 1.13, 1.14], "b": [2, 2, 3, 3]}),
            ["a", "b"],
            1,
            [
                "Excluding column: a",
                "\tOriginal dataframe: 2 duplicate rows (50.0%)",
                "\tRounded dataframe: 2 duplicate rows (50.0%)",
                "Excluding column: b",
                "\tOriginal dataframe: 0 duplicate rows (0.0%)",
                "\tRounded dataframe: 3 duplicate rows (75.0%)",
            ],
        ],
    ],
    ids=[
        "Single column",
        "Multiple columns",
        "Single column, duplicates after rounding",
        "Multiple columns, duplicates after rounding",
    ],
)
def test_check_duplicates(df, columns, round_precision, expected_printouts, capsys):
    # Act
    check_duplicates(df, columns, round_precision)

    # Assert
    actual_printout = capsys.readouterr().out
    for expected_printout in expected_printouts:
        assert expected_printout in actual_printout
