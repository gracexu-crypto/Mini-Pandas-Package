from pathlib import Path
import pandas as pd

from src.core.dataframe import DataFrame


DATA_PATH = Path(__file__).resolve().parents[1] / "Data" / "train.csv"


def read_pandas_titanic():
    return pd.read_csv(DATA_PATH)


def read_titanic():
    return DataFrame.from_csv(DATA_PATH)


def assert_close(result, expected):
    assert abs(result - expected) < 1e-9


def test_groupby_size_basic():
    df = DataFrame({
        "Team": ["A", "A", "B", "B", "B"],
        "Score": [10, 20, 30, 40, 50]
    })

    result = df.groupby("Team").size()

    assert result.to_dict() == {
        "Team": ["A", "B"],
        "size": [2, 3]
    }

    print("GroupBy size basic test passed")


def test_groupby_sum_basic():
    df = DataFrame({
        "Team": ["A", "A", "B", "B"],
        "Score": [10, 20, 30, 40]
    })

    result = df.groupby("Team").sum("Score")

    assert result.to_dict() == {
        "Team": ["A", "B"],
        "Score": [30, 70]
    }

    print("GroupBy sum basic test passed")


def test_groupby_mean_basic():
    df = DataFrame({
        "Team": ["A", "A", "B", "B"],
        "Score": [10, 20, 30, 50]
    })

    result = df.groupby("Team").mean("Score")

    assert result.to_dict() == {
        "Team": ["A", "B"],
        "Score": [15.0, 40.0]
    }

    print("GroupBy mean basic test passed")


def test_groupby_survived_mean_matches_pandas():
    df = read_titanic()
    expected = read_pandas_titanic()

    result = df.groupby("Sex").mean("Survived")
    expected_result = expected.groupby("Sex")["Survived"].mean()

    for i in range(len(result)):
        group = result["Sex"].iloc(i)
        value = result["Survived"].iloc(i)

        assert_close(value, expected_result[group])

    print("GroupBy survived mean matches pandas test passed")


def test_groupby_fare_mean_matches_pandas():
    df = read_titanic()
    expected = read_pandas_titanic()

    result = df.groupby("Pclass").mean("Fare")
    expected_result = expected.groupby("Pclass")["Fare"].mean()

    for i in range(len(result)):
        group = int(result["Pclass"].iloc(i))
        value = result["Fare"].iloc(i)

        assert_close(value, expected_result[group])

    print("GroupBy fare mean matches pandas test passed")


def test_groupby_count_matches_pandas():
    df = read_titanic()
    expected = pd.read_csv(DATA_PATH, dtype=str, keep_default_na=False)

    result = df.groupby("Sex").count("Age")
    expected_result = expected[expected["Age"] != ""].groupby("Sex")["Age"].count()

    for i in range(len(result)):
        group = result["Sex"].iloc(i)
        value = result["Age"].iloc(i)

        assert value == expected_result[group]

    print("GroupBy count matches pandas test passed")
