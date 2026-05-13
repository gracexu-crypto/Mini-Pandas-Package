import pandas as pd

from src.core.dataframe import DataFrame
from src.core.merge import merge


def assert_same_frame(result, expected):
    assert result.columns == list(expected.columns)
    assert len(result) == len(expected)
    assert result.shape == expected.shape

    for row_pos in range(len(expected)):
        for column in expected.columns:
            result_value = result[column].iloc(row_pos)
            expected_value = expected.iloc[row_pos][column]

            if pd.isna(expected_value):
                assert result_value is None
            else:
                assert result_value == expected_value


def test_merge_inner_matches_pandas():
    left_data = {
        "ID": [1, 2, 3],
        "Name": ["Alice", "Bob", "Charlie"]
    }
    right_data = {
        "ID": [2, 3, 4],
        "Score": [85, 95, 100]
    }

    left = DataFrame(left_data)
    right = DataFrame(right_data)
    result = merge(left, right, on="ID")
    expected = pd.merge(pd.DataFrame(left_data), pd.DataFrame(right_data), on="ID", how="inner")

    assert_same_frame(result, expected)

    print("Inner merge matches pandas test passed")


def test_dataframe_merge_inner_matches_pandas():
    left_data = {
        "ID": [1, 2, 3],
        "Name": ["Alice", "Bob", "Charlie"]
    }
    right_data = {
        "ID": [1, 2, 3],
        "Score": [90, 85, 95]
    }

    left = DataFrame(left_data)
    right = DataFrame(right_data)
    result = left.merge(right, on="ID")
    expected = pd.merge(pd.DataFrame(left_data), pd.DataFrame(right_data), on="ID", how="inner")

    assert_same_frame(result, expected)

    print("DataFrame merge method matches pandas test passed")


def test_merge_left_matches_pandas():
    left_data = {
        "ID": [1, 2, 3],
        "Name": ["Alice", "Bob", "Charlie"]
    }
    right_data = {
        "ID": [1, 3],
        "Score": [90, 95]
    }

    left = DataFrame(left_data)
    right = DataFrame(right_data)
    result = left.merge(right, on="ID", how="left")
    expected = pd.merge(pd.DataFrame(left_data), pd.DataFrame(right_data), on="ID", how="left")

    assert_same_frame(result, expected)

    print("Left merge matches pandas test passed")


def test_merge_with_overlapping_columns_matches_pandas():
    left_data = {
        "ID": [1, 2],
        "Value": ["left-a", "left-b"]
    }
    right_data = {
        "ID": [1, 2],
        "Value": ["right-a", "right-b"]
    }

    left = DataFrame(left_data)
    right = DataFrame(right_data)
    result = merge(left, right, on="ID")
    expected = pd.merge(pd.DataFrame(left_data), pd.DataFrame(right_data), on="ID", how="inner")

    assert_same_frame(result, expected)

    print("Merge overlapping columns matches pandas test passed")


def test_merge_with_custom_suffixes_matches_pandas():
    left_data = {
        "ID": [1, 2],
        "Value": ["left-a", "left-b"]
    }
    right_data = {
        "ID": [1, 2],
        "Value": ["right-a", "right-b"]
    }

    left = DataFrame(left_data)
    right = DataFrame(right_data)
    suffixes = ("_left", "_right")
    result = left.merge(right, on="ID", suffixes=suffixes)
    expected = pd.merge(
        pd.DataFrame(left_data),
        pd.DataFrame(right_data),
        on="ID",
        how="inner",
        suffixes=suffixes
    )

    assert_same_frame(result, expected)

    print("Merge custom suffixes matches pandas test passed")


def test_merge_multiple_keys_matches_pandas():
    left_data = {
        "ID": [1, 1, 2, 2],
        "Term": ["Fall", "Spring", "Fall", "Spring"],
        "Name": ["Alice", "Alice", "Bob", "Bob"]
    }
    right_data = {
        "ID": [1, 2, 2],
        "Term": ["Fall", "Fall", "Spring"],
        "Score": [90, 85, 88]
    }

    left = DataFrame(left_data)
    right = DataFrame(right_data)
    result = merge(left, right, on=["ID", "Term"])
    expected = pd.merge(pd.DataFrame(left_data), pd.DataFrame(right_data), on=["ID", "Term"], how="inner")

    assert_same_frame(result, expected)

    print("Merge multiple keys matches pandas test passed")


def test_merge_one_to_many_matches_pandas():
    left_data = {
        "ID": [1, 2],
        "Name": ["Alice", "Bob"]
    }
    right_data = {
        "ID": [1, 1, 2],
        "Course": ["Math", "CS", "History"]
    }

    left = DataFrame(left_data)
    right = DataFrame(right_data)
    result = merge(left, right, on="ID")
    expected = pd.merge(pd.DataFrame(left_data), pd.DataFrame(right_data), on="ID", how="inner")

    assert_same_frame(result, expected)

    print("One-to-many merge matches pandas test passed")


def test_merge_invalid_column():
    left = DataFrame({
        "ID": [1, 2],
        "Name": ["Alice", "Bob"]
    })
    right = DataFrame({
        "StudentID": [1, 2],
        "Score": [90, 85]
    })

    try:
        merge(left, right, on="ID")
    except KeyError:
        assert True
    else:
        assert False

    print("Merge invalid column test passed")


def test_merge_invalid_how():
    left = DataFrame({
        "ID": [1, 2],
        "Name": ["Alice", "Bob"]
    })
    right = DataFrame({
        "ID": [1, 2],
        "Score": [90, 85]
    })

    try:
        merge(left, right, on="ID", how="right")
    except ValueError:
        assert True
    else:
        assert False

    print("Merge invalid how test passed")
