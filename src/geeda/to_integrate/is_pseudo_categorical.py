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
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from mushu.gen_eda.is_categorical import is_categorical


def is_pseudo_categorical(
    df: pd.DataFrame,
    column: str,
    upper_threshold: float = 0.3,
    dropna: bool = True,
    nbins: int = 0,
) -> bool:
    """
    Determine if the target column is pseudo-categorical, True if the ratio of unique bins is less than the
    upper_threshold.

    :param df:
        input DataFrame
    :param column:
        target column to analyze
    :param upper_threshold:
        maximum allowed threshold for ratio of unique to total bins to be considered pseudo-categorical
    :param dropna:
        if True, drop na values from target column during analysis, else keep na values
    :param nbins:
        number of equally-sized bins to split the target column data into, default is the length of the column
    :return:
        True if the column is pseudo-categorical, False otherwise
    """
    # Sanity checks
    if df.empty:
        raise ValueError("DataFrame must not be empty")

    if column not in df.columns:
        raise KeyError(f'The column "{column}" is not in the DataFrame')

    if upper_threshold > 1 or upper_threshold <= 0:
        raise ValueError("Threshold must be a positive value less than or equal to 1")

    if not is_numeric_dtype(df[column]):
        raise TypeError("The target column must be of a numeric datatype")

    if (not isinstance(nbins, int)) or nbins < 0:
        raise TypeError("nbins must be a positive integer")

    target = df[column]

    # Default settings for bin number
    if nbins == 0:
        nbins = target.size

    # Replace inf values with NaN
    target = target.replace([np.inf, -np.inf], np.nan)

    # Generate binned column
    df["binned"] = pd.cut(target, nbins, include_lowest=True, precision=1)
    return is_categorical(df, "binned", upper_threshold, dropna)
