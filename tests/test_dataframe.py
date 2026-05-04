from pathlib import Path

import pandas as pd

from src.core.dataframe import DataFrame
from src.core.series import Series


DATA_PATH = Path(__file__).resolve().parents[1] / "Data" / "train.csv"


def read_pandas_titanic():
    return pd.read_csv(DATA_PATH, dtype=str, keep_default_na=False)


def read_titanic():
    return DataFrame.from_csv(DATA_PATH)


def assert_same_rows(result, expected, columns):
    assert len(result) == len(expected)

    for column in columns:
        assert column in result.columns

    if len(expected) > 0:
        for column in columns:
            assert result[column].iloc(0) == expected.iloc[0][column]
            assert result[column].iloc(len(expected) - 1) == expected.iloc[-1][column]


def test_dataframe_creation():
    df = DataFrame({
        "Name": ["Alice", "Bob", "Charlie"],
        "Age": [20, 21, 22]
    })

    assert len(df) == 3
    assert df.shape == (3, 2)
    assert df.columns == ["Name", "Age"]
    assert df.index == [0, 1, 2]

    print("DataFrame creation test passed")


def test_dataframe_custom_index():
    df = DataFrame({
        "Name": ["Alice", "Bob"],
        "Age": [20, 21]
    }, index=["a", "b"])

    assert df.index == ["a", "b"]
    assert df["Name"].index == ["a", "b"]
    assert df["Age"].index == ["a", "b"]

    print("DataFrame custom index test passed")


def test_dataframe_column_access():
    df = DataFrame({
        "Name": ["Alice", "Bob"],
        "Age": [20, 21]
    })

    result = df["Age"]

    assert isinstance(result, Series)
    assert result.name == "Age"
    assert list(result.data) == [20, 21]

    print("DataFrame column access test passed")


def test_dataframe_multiple_column_access():
    df = DataFrame({
        "Name": ["Alice", "Bob"],
        "Age": [20, 21],
        "Score": [90, 95]
    })

    result = df[["Name", "Score"]]

    assert isinstance(result, DataFrame)
    assert result.columns == ["Name", "Score"]
    assert result.shape == (2, 2)
    assert result.to_dict() == {
        "Name": ["Alice", "Bob"],
        "Score": [90, 95]
    }

    print("DataFrame multiple column access test passed")


def test_dataframe_invalid_column_length():
    try:
        DataFrame({
            "Name": ["Alice", "Bob"],
            "Age": [20]
        })
    except ValueError:
        assert True
    else:
        assert False

    print("DataFrame invalid column length test passed")


def test_dataframe_set_column():
    df = DataFrame({
        "Name": ["Alice", "Bob"],
        "Age": [20, 21]
    })

    df["Score"] = [90, 95]

    assert df.columns == ["Name", "Age", "Score"]
    assert list(df["Score"].data) == [90, 95]

    print("DataFrame set column test passed")


def test_dataframe_iloc_single_row():
    df = DataFrame({
        "Name": ["Alice", "Bob"],
        "Age": [20, 21]
    })

    assert df.iloc(1) == {
        "Name": "Bob",
        "Age": 21
    }

    print("DataFrame iloc single row test passed")


def test_dataframe_iloc_multiple_rows():
    df = DataFrame({
        "Name": ["Alice", "Bob", "Charlie"],
        "Age": [20, 21, 22]
    })

    result = df.iloc([0, 2])

    assert isinstance(result, DataFrame)
    assert result.index == [0, 2]
    assert result.to_dict() == {
        "Name": ["Alice", "Charlie"],
        "Age": [20, 22]
    }

    print("DataFrame iloc multiple rows test passed")


def test_dataframe_from_csv_columns_and_rows():
    df = read_titanic()
    expected = read_pandas_titanic()

    assert df.columns == list(expected.columns)
    assert len(df) == len(expected)
    assert df.shape == expected.shape

    print("CSV loading columns and rows test passed")


def test_dataframe_from_csv_basic_data():
    df = read_titanic()
    expected = read_pandas_titanic()

    columns = ["PassengerId", "Survived", "Pclass", "Name", "Sex", "Age"]

    for column in columns:
        assert df[column].iloc(0) == expected.iloc[0][column]
        assert df[column].iloc(10) == expected.iloc[10][column]
        assert df[column].iloc(len(df) - 1) == expected.iloc[-1][column]

    print("CSV loading basic data test passed")


def test_dataframe_to_csv_preserves_columns_rows_and_data(tmp_path):
    df = read_titanic()
    result = df[["PassengerId", "Survived", "Pclass", "Name", "Sex"]]
    output_path = tmp_path / "processed_results.csv"

    result.to_csv(output_path)
    expected = pd.read_csv(output_path, dtype=str, keep_default_na=False)

    assert list(expected.columns) == result.columns
    assert len(expected) == len(result)
    assert expected.iloc[0]["PassengerId"] == result["PassengerId"].iloc(0)
    assert expected.iloc[-1]["Name"] == result["Name"].iloc(len(result) - 1)

    print("CSV exporting test passed")


def test_dataframe_filter_by_sex_matches_pandas():
    df = read_titanic()
    expected = read_pandas_titanic()

    result = df[df["Sex"] == "female"]
    expected = expected[expected["Sex"] == "female"]

    assert_same_rows(result, expected, ["PassengerId", "Name", "Sex"])

    print("Boolean filtering by sex test passed")


def test_dataframe_filter_by_pclass_matches_pandas():
    df = read_titanic()
    expected = read_pandas_titanic()

    result = df[df["Pclass"] == "1"]
    expected = expected[expected["Pclass"] == "1"]

    assert_same_rows(result, expected, ["PassengerId", "Name", "Pclass"])

    print("Boolean filtering by passenger class test passed")


def test_dataframe_filter_by_age_range_matches_pandas():
    df = read_titanic()
    expected = read_pandas_titanic()
    expected_age = pd.to_numeric(expected["Age"], errors="coerce")

    mask = []
    for value in df["Age"].data:
        if value == "":
            mask.append(False)
        else:
            mask.append(18 <= float(value) <= 30)

    result = df[mask]
    expected = expected[(expected_age >= 18) & (expected_age <= 30)]

    assert_same_rows(result, expected, ["PassengerId", "Name", "Age"])

    print("Boolean filtering by age range test passed")


def test_dataframe_chained_filter_matches_pandas():
    df = read_titanic()
    expected = read_pandas_titanic()

    sex_mask = df["Sex"] == "female"
    pclass_mask = df["Pclass"] == "1"
    mask = []

    for sex_value, pclass_value in zip(sex_mask.data, pclass_mask.data):
        mask.append(bool(sex_value and pclass_value))

    result = df[mask]
    expected = expected[(expected["Sex"] == "female") & (expected["Pclass"] == "1")]

    assert_same_rows(result, expected, ["PassengerId", "Name", "Sex", "Pclass"])

    print("Chained Boolean filtering test passed")


def test_dataframe_empty_filter_result():
    df = read_titanic()

    result = df[df["Name"] == "No passenger with this name"]

    assert len(result) == 0
    assert result.shape == (0, len(df.columns))
    assert result.columns == df.columns

    print("Empty Boolean filtering result test passed")


def test_dataframe_invalid_mask_length():
    df = read_titanic()

    try:
        df[[True, False]]
    except ValueError:
        assert True
    else:
        assert False

    print("Invalid Boolean mask length test passed")
