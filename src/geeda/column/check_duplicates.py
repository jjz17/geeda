from re import I
import pandas as pd

from geeda.dataframe.geeda import Geeda
from geeda.column.is_categorical import is_categorical
from geeda.column.check_inf import check_inf
from geeda.utils import print_df


df = pd.DataFrame({"a": range(1, 6), "b": range(6, 11)})

# Arrange
geeda = Geeda(df)

# Act
geeda.apply([is_categorical, check_inf], ["a", "b"])
