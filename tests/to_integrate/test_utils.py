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

from mushu.gen_eda.utils import (
    get_list,
    validate_dataframe_columns,
    validate_no_nulls,
)


@pytest.mark.parametrize(
    "argument, expected",
    [
        ["foo", ["foo"]],
        [["foo"], ["foo"]],
        [[1, 2, 3], [1, 2, 3]],
        [[1.1, 1.2, 1.3], [1.1, 1.2, 1.3]],
    ],
    ids=[
        "Single string",
        "Single string wrapped in list",
        "List of ints",
        "List of floats",
    ],
)
def test_get_list(argument, expected):
    """
    Tests that the function make_list works as expected.
    """
    # Act, Assert
    assert get_list(argument) == expected


@pytest.mark.parametrize(
    "df, columns, expected_exception, expected_message",
    [
        [
            pd.DataFrame({"a": [1, 2], "b": [3, 4]}),
            "c",
            KeyError,
            r".*The column\(s\).*c",
        ],
        [
            pd.DataFrame({"a": [1, 2], "b": [3, 4]}),
            ["c", "d"],
            KeyError,
            r".*The column\(s\).*c.*d",
        ],
    ],
    ids=["Missing 1 column", "Missing multiple columns"],
)
def test_validate_dataframe_columns(df, columns, expected_exception, expected_message):
    """
    Tests that the function validate_dataframe_columns raises exceptions as expected.
    """

    with pytest.raises(expected_exception, match=expected_message):
        validate_dataframe_columns(df, columns)


@pytest.mark.parametrize(
    "arguments, expected_exception, expected_message",
    [
        [{"df": pd.DataFrame()}, TypeError, r"df"],
        [{"a": "Value", "b": 10, "c": None}, TypeError, r"c"],
        [
            {"df": pd.DataFrame(), "a": None, "b": None},
            TypeError,
            r".*df.*a.*b",
        ],
    ],
    ids=[
        "Empty dataframe",
        "Single null value",
        "Multiple null values and empty dataframe",
    ],
)
def test_validate_no_nulls(arguments, expected_exception, expected_message):
    """
    Tests that the function validate_no_nulls raises exceptions as expected.
    """

    with pytest.raises(expected_exception, match=expected_message):
        validate_no_nulls(arguments)
