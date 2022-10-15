import pandas as pd
from scipy import stats
from typing import Tuple


def check_distribution(column: pd.Series) -> str:
    # Use stats.kstest() for uniform, shapiro_wilk for normal
    mean = column.mean()
    std = column.std()
    size = column.size

    # Normality Test

    # Uniformity Test
    uniform = stats.uniform.rvs(loc=mean, scale=std, size=size, random_state=None)
    stats.kstest(rvs=column, cdf=uniform)
    # stats.kstest(rvs=column, cdf="uniform")
    pass
