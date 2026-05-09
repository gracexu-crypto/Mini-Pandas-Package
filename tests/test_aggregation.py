from pathlib import Path

import pandas as pd

from src.core.dataframe import DataFrame
from src.core.series import Series


DATA_PATH = Path(__file__).resolve().parents[1] / "Data" / "train.csv"


def read_pandas_titanic():
    return pd.read_csv(DATA_PATH)


def read_titanic():
    return DataFrame.from_csv(DATA_PATH)


def numeric_series(df, column, dtype=float):
    values = []

    for value in df[column].data:
        if value != "":
            values.append(dtype(value))

    return Series(values, name=column)


def assert_close(result, expected):
    assert abs(result - expected) < 1e-9


def test_series_basic_aggregation():
    s = Series([2, 4, 6, 8], name="Numbers")

    assert s.sum() == 20
    assert s.mean() == 5
    assert s.count() == 4
    assert s.min() == 2
    assert s.max() == 8

    print("Series basic aggregation test passed")


def test_age_mean_matches_pandas():
    df = read_titanic()
    expected = read_pandas_titanic()
    s = numeric_series(df, "Age")

    result = s.mean()

    assert_close(result, expected["Age"].mean())

    print("Age mean aggregation test passed")


def test_age_count_matches_pandas():
    df = read_titanic()
    expected = read_pandas_titanic()
    s = numeric_series(df, "Age")

    result = s.count()

    assert result == expected["Age"].count()

    print("Age count aggregation test passed")


def test_fare_sum_matches_pandas():
    df = read_titanic()
    expected = read_pandas_titanic()
    s = numeric_series(df, "Fare")

    result = s.sum()

    assert_close(result, expected["Fare"].sum())

    print("Fare sum aggregation test passed")


def test_fare_min_matches_pandas():
    df = read_titanic()
    expected = read_pandas_titanic()
    s = numeric_series(df, "Fare")

    result = s.min()

    assert_close(result, expected["Fare"].min())

    print("Fare min aggregation test passed")


def test_fare_max_matches_pandas():
    df = read_titanic()
    expected = read_pandas_titanic()
    s = numeric_series(df, "Fare")

    result = s.max()

    assert_close(result, expected["Fare"].max())

    print("Fare max aggregation test passed")


def test_survived_sum_matches_pandas():
    df = read_titanic()
    expected = read_pandas_titanic()
    s = numeric_series(df, "Survived", int)

    result = s.sum()

    assert result == expected["Survived"].sum()

    print("Survived sum aggregation test passed")


def test_survived_count_matches_pandas():
    df = read_titanic()
    expected = read_pandas_titanic()
    s = numeric_series(df, "Survived", int)

    result = s.count()

    assert result == expected["Survived"].count()

    print("Survived count aggregation test passed")
