import pandas as pd
import numpy as np
import scipy.stats as stats
import pytest

from mushu.gen_eda.check_distribution import (
    check_significance,
    check_normal,
    check_uniform,
    check_distribution,
    count_normal_outliers,
)

rs = np.random.RandomState(10)


@pytest.mark.parametrize(
    "p, alpha, expected_printout",
    [
        [0.01, 0.05, "Significant at the 95.0% CL"],
        [0.1, 0.05, ""],
        [0.001, None, "rejected at the 99.7% CL"],
        [0.01, None, "rejected at the 95% CL"],
        [0.3, None, "rejected at the 68% CL"],
        [0.9, None, ""],
    ],
    ids=[
        "Significant at 0.05 alpha",
        "Not significant at 0.05 alpha",
        "Significant at 99.7% CL, no alpha specified",
        "Significant at 95% CL, no alpha specified",
        "Significant at 68% CL, no alpha specified",
        "Not significant at any default alpha",
    ],
)
def test_check_significance(p, alpha, expected_printout, capsys):
    # Act
    _ = check_significance(p, alpha)

    # Assert
    assert expected_printout in capsys.readouterr().out


@pytest.mark.parametrize(
    "df, column, alpha, expected_p_value",
    [
        [
            pd.DataFrame({"a": stats.norm.rvs(size=10, random_state=rs)}),
            "a",
            0.05,
            0.7977,
        ],
        [
            pd.DataFrame(
                {"a": stats.norm(loc=10, scale=5).rvs(size=10, random_state=rs)}
            ),
            "a",
            0.05,
            0.2395,
        ],
        [
            pd.DataFrame({"a": stats.norm.rvs(size=100, random_state=rs)}),
            "a",
            0.05,
            0.5095,
        ],
        [
            pd.DataFrame(
                {"a": stats.norm(loc=20, scale=30).rvs(size=100, random_state=rs)}
            ),
            "a",
            0.05,
            0.9861,
        ],
    ],
    ids=[
        "Standard normal with 10 samples",
        "Normal with mean 10, std 5 with 10 samples",
        "Standard normal with 100 samples",
        "Normal with mean 20, std 30 with 100 samples",
    ],
)
def test_check_normal_normals(df, column, alpha, expected_p_value):
    # Act, Assert
    assert check_normal(df, column, alpha) == expected_p_value


@pytest.mark.parametrize(
    "df, column, alpha, expected_p_value",
    [
        [
            pd.DataFrame({"a": stats.uniform.rvs(size=500, random_state=rs)}),
            "a",
            0.05,
            0.0022,
        ],
        [
            pd.DataFrame({"a": stats.uniform.rvs(size=1000, random_state=rs)}),
            "a",
            0.05,
            0,
        ],
        [
            pd.DataFrame(
                {"a": stats.skewnorm.rvs(a=-20, loc=10, size=100, random_state=rs)}
            ),
            "a",
            0.05,
            0.0385,
        ],
        [
            pd.DataFrame(
                {"a": stats.skewnorm.rvs(a=20, loc=10, size=200, random_state=rs)}
            ),
            "a",
            0.05,
            0.1207,
        ],
    ],
    ids=[
        "Uniform with 500 samples",
        "Uniform with 1000 samples",
        "Left skew with 100 samples",
        "Right skew with 200 samples",
    ],
)
def test_check_normal_non_normals(df, column, alpha, expected_p_value):
    # Act, Assert
    assert check_normal(df, column, alpha) == expected_p_value


@pytest.mark.parametrize(
    "df, column, alpha, expected_p_value",
    [
        [
            pd.DataFrame({"a": stats.uniform.rvs(size=50, random_state=rs)}),
            "a",
            0.05,
            0.4495,
        ],
        [
            pd.DataFrame({"a": stats.uniform.rvs(size=100, random_state=rs)}),
            "a",
            0.05,
            0.7124,
        ],
        [
            pd.DataFrame({"a": stats.uniform.rvs(size=500, random_state=rs)}),
            "a",
            0.05,
            0.1446,
        ],
        [
            pd.DataFrame({"a": stats.uniform.rvs(size=1000, random_state=rs)}),
            "a",
            0.05,
            0.0844,
        ],
    ],
    ids=[
        "Uniform with 50 samples",
        "Uniform with 100 samples",
        "Uniform with 500 samples",
        "Uniform with 1000 samples",
    ],
)
def test_check_uniform_uniforms(df, column, alpha, expected_p_value):
    # Act, Assert
    assert check_uniform(df, column, alpha) == expected_p_value


@pytest.mark.parametrize(
    "df, column, alpha, expected_p_value",
    [
        [
            pd.DataFrame(
                {"a": stats.skewnorm.rvs(a=-20, loc=10, size=100, random_state=rs)}
            ),
            "a",
            0.05,
            0,
        ],
        [
            pd.DataFrame(
                {"a": stats.skewnorm.rvs(a=20, loc=10, size=200, random_state=rs)}
            ),
            "a",
            0.05,
            0,
        ],
        [
            pd.DataFrame({"a": stats.norm.rvs(size=30, random_state=rs)}),
            "a",
            0.05,
            0,
        ],
        [
            pd.DataFrame(
                {"a": stats.norm(loc=10, scale=5).rvs(size=100, random_state=rs)}
            ),
            "a",
            0.05,
            0.0005,
        ],
        [
            pd.DataFrame({"a": stats.norm.rvs(size=100, random_state=rs)}),
            "a",
            0.05,
            0,
        ],
        [
            pd.DataFrame(
                {"a": stats.norm(loc=20, scale=30).rvs(size=200, random_state=rs)}
            ),
            "a",
            0.05,
            0,
        ],
    ],
    ids=[
        "Left skew with 100 samples",
        "Right skew with 200 samples",
        "Standard normal with 30 samples",
        "Normal with mean 10, std 5 with 100 samples",
        "Standard normal with 100 samples",
        "Normal with mean 20, std 30 with 200 samples",
    ],
)
def test_check_uniform_non_uniforms(df, column, alpha, expected_p_value):
    # Act, Assert
    assert check_uniform(df, column, alpha) == expected_p_value


@pytest.mark.parametrize(
    "df, column, alpha, expected_count",
    [
        [
            pd.DataFrame({"a": stats.uniform.rvs(size=50, random_state=rs)}),
            "a",
            0.05,
            49,
        ],
        [
            pd.DataFrame({"a": stats.uniform.rvs(size=100, random_state=rs)}),
            "a",
            0.05,
            98,
        ],
        [
            pd.DataFrame({"a": stats.uniform.rvs(size=500, random_state=rs)}),
            "a",
            0.05,
            490,
        ],
        [
            pd.DataFrame({"a": stats.uniform.rvs(size=1000, random_state=rs)}),
            "a",
            0.05,
            979,
        ],
        [
            pd.DataFrame(
                {"a": stats.skewnorm.rvs(a=-20, loc=10, size=100, random_state=rs)}
            ),
            "a",
            0.05,
            97,
        ],
        [
            pd.DataFrame(
                {"a": stats.skewnorm.rvs(a=20, loc=10, size=200, random_state=rs)}
            ),
            "a",
            0.05,
            190,
        ],
        [pd.DataFrame({"a": stats.norm.rvs(size=30, random_state=rs)}), "a", 0.05, 29],
        [
            pd.DataFrame(
                {"a": stats.norm(loc=10, scale=5).rvs(size=100, random_state=rs)}
            ),
            "a",
            0.05,
            96,
        ],
        [pd.DataFrame({"a": stats.norm.rvs(size=100, random_state=rs)}), "a", 0.05, 92],
        [
            pd.DataFrame(
                {"a": stats.norm(loc=20, scale=30).rvs(size=200, random_state=rs)}
            ),
            "a",
            0.05,
            192,
        ],
    ],
    ids=[
        "Uniform with 50 samples",
        "Uniform with 100 samples",
        "Uniform with 500 samples",
        "Uniform with 1000 samples",
        "Left skew with 100 samples",
        "Right skew with 200 samples",
        "Standard normal with 30 samples",
        "Normal with mean 10, std 5 with 100 samples",
        "Standard normal with 100 samples",
        "Normal with mean 20, std 30 with 200 samples",
    ],
)
def test_count_outliers(df, column, alpha, expected_count):
    # Act, Assert
    assert count_normal_outliers(df, column, alpha) == expected_count


@pytest.mark.parametrize(
    "df, column, alpha, expected_classification",
    [
        [
            pd.DataFrame({"a": stats.uniform.rvs(size=500, random_state=rs)}),
            "a",
            0.05,
            "Uniform",
        ],
        [
            pd.DataFrame({"a": stats.uniform.rvs(size=750, random_state=rs)}),
            "a",
            0.05,
            "Uniform",
        ],
        [
            pd.DataFrame({"a": stats.uniform.rvs(size=1000, random_state=rs)}),
            "a",
            0.05,
            "Uniform",
        ],
        [
            pd.DataFrame(
                {"a": stats.skewnorm.rvs(a=-20, loc=10, size=400, random_state=rs)}
            ),
            "a",
            0.05,
            "Unknown",
        ],
        [
            pd.DataFrame(
                {"a": stats.skewnorm.rvs(a=20, loc=10, size=500, random_state=rs)}
            ),
            "a",
            0.05,
            "Unknown",
        ],
        [
            pd.DataFrame({"a": stats.poisson.rvs(mu=2, size=1000, random_state=rs)}),
            "a",
            0.05,
            "Unknown",
        ],
        [
            pd.DataFrame({"a": stats.norm.rvs(size=30, random_state=rs)}),
            "a",
            0.05,
            "Normal",
        ],
        [
            pd.DataFrame(
                {"a": stats.norm(loc=10, scale=5).rvs(size=50, random_state=rs)}
            ),
            "a",
            0.05,
            "Normal",
        ],
        [
            pd.DataFrame({"a": stats.norm.rvs(size=100, random_state=rs)}),
            "a",
            0.05,
            "Normal",
        ],
        [
            pd.DataFrame(
                {"a": stats.norm(loc=20, scale=30).rvs(size=100, random_state=rs)}
            ),
            "a",
            0.05,
            "Normal",
        ],
    ],
    ids=[
        "Uniform with 500 samples",
        "Uniform with 750 samples",
        "Uniform with 1000 samples",
        "Left skew with 400 samples",
        "Right skew with 500 samples",
        "Poisson with mu 2 with 1000 samples",
        "Standard normal with 30 samples",
        "Normal with mean 10, std 5 with 50 samples",
        "Standard normal with 100 samples",
        "Normal with mean 20, std 30 with 100 samples",
    ],
)
def test_check_distribution(df, column, alpha, expected_classification):
    # Act, Assert
    assert check_distribution(df, column, alpha) == expected_classification
