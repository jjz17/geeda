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


def is_categorical(
    df: pd.DataFrame, column: str, upper_threshold: float = 0.3, dropna: bool = True
) -> bool:
    """
    Determine if the target column is categorical, True if the ratio of unique values is less than the
    upper_threshold.

    :param df:
        input DataFrame
    :param column:
        target column to analyze
    :param upper_threshold:
        maximum allowed threshold for ratio of unique to total values to be considered categorical
    :param dropna:
        if True, drop na values from target column during analysis, else keep na values
    :return:
        True if the column is categorical, False otherwise
    """
    # Sanity checks
    if df.empty:
        raise ValueError("DataFrame must not be empty")

    if column not in df.columns:
        raise KeyError(f'The column "{column}" is not in the DataFrame')

    if upper_threshold > 1 or upper_threshold <= 0:
        raise ValueError("Threshold must be a positive value less than or equal to 1")

    target = df[column]

    if dropna:
        target = target.dropna()

    unique = target.nunique()
    rows = target.size
    return (unique / rows) < upper_threshold
