class GroupBy:
    def __init__(self, frame, by):
        self.frame = frame
        self.by = [by] if isinstance(by, str) else list(by)
        self.groups = self._make_groups()

    def _make_groups(self):
        groups = {}

        for row_pos in range(len(self.frame.index)):
            key = self._make_key(row_pos)

            if key not in groups:
                groups[key] = []

            groups[key].append(row_pos)

        return groups

    def _make_key(self, row_pos):
        values = []

        for column in self.by:
            if column not in self.frame.data:
                raise KeyError(column)

            values.append(self.frame.data[column].data[row_pos])

        if len(values) == 1:
            return values[0]

        return tuple(values)

    def _new_data(self):
        data = {}

        for column in self.by:
            data[column] = []

        return data

    def _append_key(self, data, key):
        if len(self.by) == 1:
            data[self.by[0]].append(key)
            return

        for column, value in zip(self.by, key):
            data[column].append(value)

    def _columns(self, columns):
        if columns is None:
            return [column for column in self.frame.columns if column not in self.by]

        if isinstance(columns, str):
            return [columns]

        return list(columns)

    def _values(self, column, rows):
        values = []

        for row_pos in rows:
            value = self.frame.data[column].data[row_pos]
            if value is not None and value != "":
                values.append(value)

        return values

    def _numeric_values(self, column, rows):
        values = []

        for value in self._values(column, rows):
            values.append(float(value))

        return values

    def _format_number(self, value):
        if isinstance(value, float) and value.is_integer():
            return int(value)

        return value

    def get_group(self, key):
        if key not in self.groups:
            raise KeyError(key)

        return self.frame.iloc(self.groups[key])

    def size(self):
        data = self._new_data()
        data["size"] = []

        for key, rows in self.groups.items():
            self._append_key(data, key)
            data["size"].append(len(rows))

        return self.frame.__class__(data)

    def count(self, columns=None):
        columns = self._columns(columns)
        data = self._new_data()

        for column in columns:
            data[column] = []

        for key, rows in self.groups.items():
            self._append_key(data, key)

            for column in columns:
                data[column].append(len(self._values(column, rows)))

        return self.frame.__class__(data)

    def sum(self, columns):
        return self._aggregate(columns, "sum")

    def mean(self, columns):
        return self._aggregate(columns, "mean")

    def min(self, columns):
        return self._aggregate(columns, "min")

    def max(self, columns):
        return self._aggregate(columns, "max")

    def agg(self, columns, func):
        return self._aggregate(columns, func)

    def aggregate(self, columns, func):
        return self.agg(columns, func)

    def _aggregate(self, columns, func):
        columns = self._columns(columns)
        data = self._new_data()

        for column in columns:
            data[column] = []

        for key, rows in self.groups.items():
            self._append_key(data, key)

            for column in columns:
                values = self._numeric_values(column, rows)
                data[column].append(self._apply_func(values, func))

        return self.frame.__class__(data)

    def _apply_func(self, values, func):
        if len(values) == 0:
            return None

        if func == "sum":
            return self._format_number(sum(values))

        if func == "mean":
            return sum(values) / len(values)

        if func == "min":
            return self._format_number(min(values))

        if func == "max":
            return self._format_number(max(values))

        if callable(func):
            return func(values)

        raise ValueError("unsupported aggregation function")
