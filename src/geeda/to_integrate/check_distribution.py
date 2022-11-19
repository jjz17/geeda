from typing import Optional

import pandas as pd
import numpy as np

from scipy import stats


def check_significance(
    p: float, alpha: Optional[float] = None, decimals: int = 4
) -> float:
    """
    Generates printout of statistical test significance at a given or default alpha level.

    :param p:
        p-value of statistical test
    :param alpha:
        Alpha level for statistical test, defaults are 0.003, 0.05, and 0.32
    :param decimals:
        Number of decimals to round p-value to, default is 4
    :return:
        p-value of statistical test
    """
    if alpha is not None and p <= alpha:
        print(f"Significant at the {round((1 - alpha)*100, 1)}% CL")
    else:
        if p <= 1 - 0.997:
            print("Null hypothesis is rejected at the 99.7% CL")
        elif p <= 1 - 0.95:
            print("Null hypothesis is rejected at the 95% CL")
        elif p <= 1 - 0.68:
            print("Null hypothesis is rejected at the 68% CL")

    return round(p, decimals)


def check_normal(df: pd.DataFrame, column: str, alpha: float = 0.05) -> float:
    """
     Check if data is distributed normally, uses Shapiro-Wilk normality test for small sample sizes (< 50),
     Kolmogorov-Smirnov goodness-of-fit test against a normal distribution for large sample sizes.

    :param df:
         input DataFrame
     :param column:
         target column to analyze
     :param alpha:
         Alpha level for statistical test, default is 0.05
     :return:
         p-value of statistical test
    """
    data = df[column]

    if data.size < 50:
        test_statistic, p = stats.shapiro(data)
    else:
        test_statistic, p = stats.kstest(
            data, stats.norm(loc=data.mean(), scale=data.std()).cdf
        )

    return check_significance(p, alpha)


def check_uniform(df: pd.DataFrame, column: str, alpha: float = 0.05) -> float:
    """
     Check if data is distributed uniformly, uses Kolmogorov-Smirnov goodness-of-fit test
     against the uniform distribution.

    :param df:
         input DataFrame
     :param column:
         target column to analyze
     :param alpha:
         Alpha level for statistical test, default is 0.05
     :return:
         p-value of statistical test
    """
    data = df[column]

    uniform = stats.uniform(loc=data.min(), scale=data.max())
    test_statistic, p = stats.kstest(data, uniform.cdf)

    return check_significance(p, alpha)


def count_normal_outliers(
    df: pd.DataFrame, column: str, z_score_threshold: float = 3.0
) -> int:
    """
     Count the number of outliers in the data, assuming the distribution is normal.

    :param df:
         input DataFrame
     :param column:
         target column to analyze
     :param z_score_threshold:
         The maximum absolute-value z-score before data are considered as outliers
     :return:
         The number of outliers in the data
    """
    data = df[column]

    return data[np.abs(stats.zscore(data)) > z_score_threshold].size


def check_distribution(
    df: pd.DataFrame, column: str, alpha: float = 0.05, z_score_threshold: float = 3.0
) -> str:
    """
     Identify if the distribution of the data is normal or uniform and count number of outliers.

    :param df:
         input DataFrame
     :param column:
         target column to analyze
     :param alpha:
         Alpha level for statistical tests, default is 0.05
     :param z_score_threshold:
         The maximum absolute-value z-score before data are considered as outliers
     :return:
         The distribution type of the data, `Unknown` if not normal or uniform
    """

    distribution = "Unknown"
    outliers = count_normal_outliers(df, column, z_score_threshold)
    norm_p = check_normal(df, column, alpha)
    uniform_p = check_uniform(df, column, alpha)

    printout = f"Normality test: p={norm_p}\nUniformity test: p={uniform_p}\nOutliers: {outliers}"
    print(printout)

    if norm_p > alpha:
        distribution = "Normal"
    elif uniform_p > alpha:
        distribution = "Uniform"
    return distribution
