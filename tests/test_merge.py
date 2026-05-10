from src.core.dataframe import DataFrame


def test_merge_basic():
    left = DataFrame({
        "ID": [1, 2, 3],
        "Name": ["Alice", "Bob", "Charlie"]
    })

    right = DataFrame({
        "ID": [1, 2, 3],
        "Score": [90, 85, 95]
    })

    result = left.merge(right, on="ID")

    assert result.to_dict() == {
        "ID": [1, 2, 3],
        "Name": ["Alice", "Bob", "Charlie"],
        "Score": [90, 85, 95]
    }

    print("Basic merge test passed")


def test_merge_only_matching_rows():
    left = DataFrame({
        "ID": [1, 2, 3],
        "Name": ["Alice", "Bob", "Charlie"]
    })

    right = DataFrame({
        "ID": [2, 3, 4],
        "Score": [85, 95, 100]
    })

    result = left.merge(right, on="ID")

    assert result.to_dict() == {
        "ID": [2, 3],
        "Name": ["Bob", "Charlie"],
        "Score": [85, 95]
    }

    print("Merge only matching rows test passed")


def test_merge_with_extra_columns():
    left = DataFrame({
        "ID": [1, 2],
        "Name": ["Alice", "Bob"],
        "Age": [20, 21]
    })

    right = DataFrame({
        "ID": [1, 2],
        "Grade": ["A", "B"],
        "Score": [95, 88]
    })

    result = left.merge(right, on="ID")

    assert result.to_dict() == {
        "ID": [1, 2],
        "Name": ["Alice", "Bob"],
        "Age": [20, 21],
        "Grade": ["A", "B"],
        "Score": [95, 88]
    }

    print("Merge with extra columns test passed")


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
        left.merge(right, on="ID")
    except KeyError:
        assert True
    else:
        assert False

    print("Merge invalid column test passed")
