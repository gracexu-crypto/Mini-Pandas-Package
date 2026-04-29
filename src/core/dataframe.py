from .series import Series

class DataFrame:
    def __init__(self, data, index=None, copy=True):
        self.data = {}
        self.columns = list(data.keys())

        if len(self.columns) == 0:
            self.index = list(index) if index is not None else []
            return

        first_column = data[self.columns[0]]
        row_count = len(first_column)
        self.index = list(index) if index is not None else list(range(row_count))

        if len(self.index) != row_count:
            raise ValueError("data and index must have the same length")

        for column in self.columns:
            values = data[column]

            if isinstance(values, Series):
                series = values.copy() if copy else values
                if index is not None:
                    series.index = self.index.copy()
                    series._index_map = series.index_map()
                if series.name is None:
                    series.name = column
            else:
                series = Series(values, index=self.index, name=column, copy=copy)

            if len(series) != row_count:
                raise ValueError("all columns must have the same length")

            if series.index != self.index:
                raise ValueError("all Series indexes must match DataFrame index")

            self.data[column] = series

    @property
    def shape(self):
        return (len(self.index), len(self.columns))

    def __len__(self):
        return len(self.index)

    def __repr__(self):
        if len(self.columns) == 0:
            return "Empty DataFrame"

        index_strings = [str(idx) for idx in self.index]
        index_width = max(len(s) for s in index_strings) if index_strings else 0

        column_widths = {}
        for column in self.columns:
            value_strings = [str(value) for value in self.data[column].data]
            value_width = max(len(s) for s in value_strings) if value_strings else 0
            column_widths[column] = max(len(str(column)), value_width)

        header_values = [" " * index_width]
        for column in self.columns:
            header_values.append(f"{str(column):<{column_widths[column]}}")

        lines = ["\t".join(header_values)]

        for row_pos, idx in enumerate(self.index):
            row_values = [f"{str(idx):<{index_width}}"]
            for column in self.columns:
                value = self.data[column].data[row_pos]
                row_values.append(f"{str(value):<{column_widths[column]}}")
            lines.append("\t".join(row_values))

        return "\n".join(lines)

    def __getitem__(self, key):
        if isinstance(key, str):
            if key not in self.data:
                raise KeyError(key)
            return self.data[key]

        if isinstance(key, list):
            new_data = {}
            for column in key:
                if column not in self.data:
                    raise KeyError(column)
                new_data[column] = self.data[column].copy()
            return DataFrame(new_data, index=self.index.copy())

        raise KeyError(key)

    def __setitem__(self, key, value):
        if isinstance(value, Series):
            series = value.copy()
            if series.index != self.index:
                raise ValueError("Series index must match DataFrame index")
            if series.name is None:
                series.name = key
        else:
            series = Series(value, index=self.index, name=key)

        if len(series) != len(self):
            raise ValueError("column length must match DataFrame length")

        if key not in self.data:
            self.columns.append(key)

        self.data[key] = series

    def iloc(self, pos):
        if isinstance(pos, int):
            return {
                column: self.data[column].data[pos]
                for column in self.columns
            }

        if isinstance(pos, list):
            new_index = [self.index[i] for i in pos]
            new_data = {}
            for column in self.columns:
                new_data[column] = [self.data[column].data[i] for i in pos]
            return DataFrame(new_data, index=new_index)

        raise TypeError("iloc only supports int or list")

    def to_dict(self):
        return {
            column: list(self.data[column].data)
            for column in self.columns
        }

    def copy(self):
        return DataFrame(
            {
                column: self.data[column].copy()
                for column in self.columns
            },
            index=self.index.copy()
        )
