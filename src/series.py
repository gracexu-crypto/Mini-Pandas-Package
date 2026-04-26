import numpy as np

class Series:
    def __init__(self, data, index=None, name=None, copy=True):
        self.data = np.array(data, copy=copy)
        self.index = list(index) if index is not None else list(range(len(data)))    
        self.name = name

         self._index_map = self.index_map()

        if len(self.data) != len(self.index):
            raise ValueError("data and index must have the same length")

        if self.data.ndim != 1:
            raise ValueError("Series data must be 1-dimensional")
    def __len__(self):
        return len(self.data)
    
    def __repr__(self):
        index_strings = [str(idx) for idx in self.index]
        value_strings = [str(value) for value in self.data]

        index_width = max(len(s) for s in index_strings) if index_strings else 0
        value_width = max(len(s) for s in value_strings) if value_strings else 0

        lines = []

        for idx_str, value_str in zip(index_strings, value_strings):
            line = f"{idx_str:<{index_width}}\t{value_str:<{value_width}}"
            lines.append(line)

        if self.name is not None:
            lines.append(f"Name: {self.name}")
        lines.append(f"dtype: {self.data.dtype}")

        return "\n".join(lines)
    
    def iloc(self, pos):
        return self.data[pos]

    def index_map(self):
        return {idx: pos for pos, idx in enumerate(self.index)}

    def loc(self, label):
        pos = self._index_map[label]
        return self.data[pos]
    
    def __getitem__(self, key):
        if isinstance(key, int):
            return self.iloc(key)

        if key in self._index_map:
            return self.loc(key)

        raise KeyError(key)
    
    def to_numpy(self):
        return self.data.copy()

    def copy(self):
        return Series(
            self.data.copy(),
            index=self.index.copy(),
            name=self.name
        )

    def _compare(self, other, op):
        result = op(self.data, other)
        return Series(result, index=self.index, name=self.name)
    
    def __eq__(self, other):
        return self._compare(other, np.equal)

    def __ne__(self, other):
        return self._compare(other, np.not_equal)

    def __gt__(self, other):
        return self._compare(other, np.greater)

    def __ge__(self, other):
        return self._compare(other, np.greater_equal)

    def __lt__(self, other):
        return self._compare(other, np.less)

    def __le__(self, other):
        return self._compare(other, np.less_equal)
    
    def sum(self):
        return self.data.sum()

    def mean(self):
        return self.data.mean()

    def min(self):
        return self.data.min()

    def max(self):
        return self.data.max()

    def count(self):
        # TODO: For now, count() returns the total number of values.
        # After implementing missing value handling and filtering,
        # update this method to count only non-NA values.
        return len(self.data)
    
    # only supports Series with scalar for now
    def arithmetic(self, other, op):
        result = op(self.data, other)
        return Series(result, index=self.index, name=self.name)

    def __add__(self, other):
        return self.arithmetic(other, np.add)

    def __sub__(self, other):
        return self.arithmetic(other, np.subtract)

    def __mul__(self, other):
        return self.arithmetic(other, np.multiply)

    def __truediv__(self, other):
        return self.arithmetic(other, np.divide)
        



    

