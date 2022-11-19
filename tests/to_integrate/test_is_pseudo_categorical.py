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

from mushu.gen_eda.is_pseudo_categorical import is_pseudo_categorical


@pytest.mark.parametrize(
    "df, column, upper_threshold, expected_classification",
    [
        # test for categorical with low upper_threshold
        [pd.DataFrame({"a": [1, 1, 1, 1, 1, 1]}), "a", 0.2, True],
        # test for categorical with high upper_threshold
        [pd.DataFrame({"a": [1, 1, 2, 2]}), "a", 0.8, True],
        # test for categorical with floats
        [pd.DataFrame({"a": [1.1, 1.11, 1.12, 1.13, 1.31, 1.32]}), "a", 0.4, True],
        # test for not categorical with low upper_threshold
        [pd.DataFrame({"a": [1, 1, 2, 2]}), "a", 0.2, False],
        # test for not categorical with high upper_threshold
        [pd.DataFrame({"a": [1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]}), "a", 0.8, False],
        # test for not categorical with floats
        [pd.DataFrame({"a": [1.1, 1.2, 1.3, 1.3, 1.3, 1.4]}), "a", 0.2, False],
    ],
)
def test_is_pseudo_categorical(df, column, upper_threshold, expected_classification):
    """
    Tests that the function is_pseudo_categorical works as expected.
    """

    # Assert
    assert is_pseudo_categorical(df, column, upper_threshold) == expected_classification


@pytest.mark.parametrize(
    "df, column, upper_threshold, dropna, expected_classification",
    [
        # test for categorical with dropna True
        [
            pd.DataFrame(
                {"a": [1.1, 1.15, 1.17, 2.5, 2.6, None, None, None, None, None]}
            ),
            "a",
            0.5,
            True,
            True,
        ],
        # test for categorical with dropna False
        [
            pd.DataFrame(
                {"a": [1.1, 1.15, 1.17, 2.5, 2.6, None, None, None, None, None]}
            ),
            "a",
            0.3,
            False,
            True,
        ],
        # test for not categorical with dropna True
        [
            pd.DataFrame(
                {"a": [1.1, 1.15, 1.17, 2.5, 2.6, None, None, None, None, None]}
            ),
            "a",
            0.4,
            True,
            False,
        ],
        # test for not categorical with dropna False
        [
            pd.DataFrame(
                {"a": [1.1, 1.15, 1.17, 2.5, 2.6, None, None, None, None, None]}
            ),
            "a",
            0.2,
            False,
            False,
        ],
    ],
)
def test_dropna_is_pseudo_categorical(
    df, column, upper_threshold, dropna, expected_classification
):
    """
    Tests that the function is_categorical works as expected.
    """

    # Assert
    assert (
        is_pseudo_categorical(df, column, upper_threshold, dropna)
        == expected_classification
    )


@pytest.mark.parametrize(
    "df, column, upper_threshold, expected_exception, expected_message",
    [
        # test for a missing column
        [
            pd.DataFrame({"a": [1, 2], "b": [3, 4]}),
            "c",
            0.2,
            KeyError,
            r'.*The column "c".*',
        ],
        # test for an empty frame
        [
            pd.DataFrame(columns=["a", "b"]),
            "a",
            0.2,
            ValueError,
            r"DataFrame must not be empty",
        ],
        # test for a low upper_threshold
        [
            pd.DataFrame({"a": [1, 2], "b": [3, 4]}),
            "a",
            0,
            ValueError,
            r".*Threshold must be .*",
        ],
        # test for a high upper_threshold
        [
            pd.DataFrame({"a": [1, 2], "b": [3, 4]}),
            "a",
            1.01,
            ValueError,
            r".*Threshold must be .*",
        ],
        # test for non-numeric column
        [
            pd.DataFrame({"a": ["1", "2"], "b": [3, 4]}),
            "a",
            0.2,
            TypeError,
            r".*numeric .*",
        ],
        # test for a None DataFrame input
        [
            None,
            "a",
            0.4,
            AttributeError,
            r".*object has no attribute.*",
        ],
    ],
)
def test_raise_exceptions(
    df, column, upper_threshold, expected_exception, expected_message
):
    """
    Tests that the function is_pseudo_categorical raises exceptions as expected.
    """

    with pytest.raises(expected_exception, match=expected_message):
        is_pseudo_categorical(df, column, upper_threshold)


@pytest.mark.parametrize(
    "df, column, upper_threshold, nbins, expected_exception, expected_message",
    [
        # test for non-integer bin input
        [
            pd.DataFrame({"a": [1, 2], "b": [3, 4]}),
            "a",
            0.2,
            0.5,
            TypeError,
            r".*nbins.*",
        ],
        # test for negative bin input
        [
            pd.DataFrame({"a": [1, 2], "b": [3, 4]}),
            "a",
            0.2,
            -1,
            TypeError,
            r".*nbins.*",
        ],
        # test for None bin input
        [
            pd.DataFrame({"a": [1, 2], "b": [3, 4]}),
            "a",
            0.2,
            None,
            TypeError,
            r".*nbins.*",
        ],
    ],
)
def test_nbins_raise_exceptions(
    df, column, upper_threshold, nbins, expected_exception, expected_message
):
    """
    Tests that the function is_pseudo_categorical raises exceptions as expected.
    """

    with pytest.raises(expected_exception, match=expected_message):
        is_pseudo_categorical(df, column, upper_threshold, nbins=nbins)
