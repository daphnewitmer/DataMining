"""
Simplest preprocessing steps to make it work with model
"""


def remove_nan_values(data):
    return data.dropna(axis=1, how="any")


def drop_columns(data, columns):
    return data.drop(columns, axis=1)


def add_target_attribute(row):
    if row.booking_bool > 0:
        return int(5)
    if row.click_bool > 0:
        return int(1)
    return int(0)
