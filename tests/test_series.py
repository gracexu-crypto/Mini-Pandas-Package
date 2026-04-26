import numpy as np
from mini_pandas import Series

def test_series_creation():
  s = Series([10, 20, 30], name="Scores")

  assert len(s) == 3
  assert s.name == "Scores"
  assert s.index == [0, 1, 2]
  assert np.array_equal(s.data, np.array([10, 20, 30]))

  print("Series creation test passes")


def test_series_custom_index():
  s = Series([85, 90, 95], index=["a", "b", "c"], name="Grades")

  assert s.index == ["a", "b", "c"]
  assert s.iloc(0) == 85
  assert s.iloc(1) == 90
  assert s.iloc(2) == 95

  print("Series custom index test passed")


def test_series_indexing():
  s = Series([5, 10, 15])

  assert s[0] == 5
  assert s[1] == 10
  assert s[2] == 15

  print("Series indexing test passed")


def test_series_math():
    s = Series([2, 4, 6]) 

 assert s.sum() == 12
    assert s.mean() == 4
    assert s.min() == 2
    assert s.max() == 6
    assert s.count() == 3

    print("Series math test passed")


def test_series_arithmetic():
    s = Series([1, 2, 3])

    added = s + 2
    multiplied = s * 3

    assert np.array_equal(added.data, np.array([3, 4, 5]))
    assert np.array_equal(multiplied.data, np.array([3, 6, 9]))

    print("Series arithmetic test passed")


def test_series_comparison():
    s = Series([10, 20, 30])

    result = s > 15

    assert np.array_equal(result.data, np.array([False, True, True]))

    print("Series comparison test passed")


test_series_creation()
test_series_custom_index()
test_series_indexing()
test_series_math()
test_series_arithmetic()
test_series_comparison()
