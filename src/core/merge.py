def merge(left, right, on, how="inner", suffixes=("_x", "_y")):
    if how not in ("inner", "left"):
        raise ValueError("merge only supports inner and left")

    on_columns = [on] if isinstance(on, str) else list(on)

    for column in on_columns:
        if column not in left.columns:
            raise KeyError(column)
        if column not in right.columns:
            raise KeyError(column)

    right_map = _make_right_map(right, on_columns)
    output_columns = _make_output_columns(left, right, on_columns, suffixes)
    new_data = {column: [] for column in output_columns}

    for left_pos in range(len(left.index)):
        key = _make_key(left, on_columns, left_pos)
        right_positions = right_map.get(key, [])

        if len(right_positions) == 0 and how == "left":
            _append_row(new_data, left, right, on_columns, suffixes, left_pos, None)

        for right_pos in right_positions:
            _append_row(new_data, left, right, on_columns, suffixes, left_pos, right_pos)

    return left.__class__(new_data)


def _make_right_map(right, on_columns):
    right_map = {}

    for right_pos in range(len(right.index)):
        key = _make_key(right, on_columns, right_pos)

        if key not in right_map:
            right_map[key] = []

        right_map[key].append(right_pos)

    return right_map


def _make_key(frame, on_columns, row_pos):
    values = []

    for column in on_columns:
        values.append(frame.data[column].data[row_pos])

    if len(values) == 1:
        return values[0]

    return tuple(values)


def _make_output_columns(left, right, on_columns, suffixes):
    output_columns = []
    overlap = _overlap_columns(left, right, on_columns)

    for column in left.columns:
        if column in overlap:
            output_columns.append(column + suffixes[0])
        else:
            output_columns.append(column)

    for column in right.columns:
        if column in on_columns:
            continue

        if column in overlap:
            output_columns.append(column + suffixes[1])
        else:
            output_columns.append(column)

    return output_columns


def _overlap_columns(left, right, on_columns):
    overlap = []

    for column in left.columns:
        if column in right.columns and column not in on_columns:
            overlap.append(column)

    return overlap


def _column_name(column, side, overlap, suffixes):
    if column in overlap:
        if side == "left":
            return column + suffixes[0]

        return column + suffixes[1]

    return column


def _append_row(new_data, left, right, on_columns, suffixes, left_pos, right_pos):
    overlap = _overlap_columns(left, right, on_columns)

    for column in left.columns:
        output_column = _column_name(column, "left", overlap, suffixes)
        new_data[output_column].append(left.data[column].data[left_pos])

    for column in right.columns:
        if column in on_columns:
            continue

        output_column = _column_name(column, "right", overlap, suffixes)

        if right_pos is None:
            new_data[output_column].append(None)
        else:
            new_data[output_column].append(right.data[column].data[right_pos])
