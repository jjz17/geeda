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
import numpy as np
from typing import List, Union, Dict

from mushu.gen_eda.utils import get_list, validate_dataframe_columns


def check_inf(
    df: pd.DataFrame,
    columns: Union[str, List[str]] = None,
) -> Dict[str, int]:
    """
    Identify if Inf values are present in given columns.

    :param df:
        input DataFrame
    :param columns:
        target column(s) to analyze, default is all columns
    :return:
        a dict with column names as keys and the count of Inf values as values
    """
    inf_counts: Dict[str, int] = dict()

    columns = df.columns if columns is None else get_list(columns)

    validate_dataframe_columns(df, columns)

    for column in columns:
        data = df[column]

        inf_counts[column] = np.isinf(data).values.sum()

    return inf_counts
