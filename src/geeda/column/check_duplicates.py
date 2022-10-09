import pandas as pd

from geeda.dataframe.geeda import Geeda
from geeda.column.is_categorical import is_categorical
from geeda.utils import print_df


df = pd.DataFrame({"a": range(1, 6), "b": range(6, 11)})

# Arrange
geeda = Geeda(df)

# Act
geeda.apply([is_categorical], ["a", "b"])


print(is_categorical(df.a))

print_df(pd.DataFrame({"a": [False, True], "b": [True, False]}))
