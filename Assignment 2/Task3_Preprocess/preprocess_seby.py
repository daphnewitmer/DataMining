import pandas as pd
import numpy as np


def data_input(path, complete=False, nrows=10000):
    """
    Function to read a datafile.
    Arguments: path, complete, nrows
    path: Filepath to the data
    complete: Read the whole dataset (True) or specific number of rows (nrows)
    nrows: Specify number of rows for input (default: 10000)
    """

    if complete:
        df = pd.read_csv(path)

    else:
        df = pd.read_csv(path, nrows=nrows)
        df["date_time"] = pd.to_datetime(
            df["date_time"], format="%Y-%m-%d %H:%M:%S")

    #Maybe we could get rid of the exact timestamp if not useful
    #-> .apply(lambda x: x.date())
    return df


def drop_attributes(df, cutoff=25, extra_add=[]):
    """
    Function that drops attributes with a % condition
    Arguments: df, cutoff, extra_add
    df: Daraframe
    cutoff: Percentage of desired cutoff from attributes (default: more than 25% missing)
    extra_add: Insert column name for manual drop a attribute (default: empty)
    """

    df_copy = df.copy()

    attributs_drop = []
    for var in sorted(df.columns):
        series = df[var]
        perc_missing = 100 - series.count() / len(series) * 100

        if perc_missing > cutoff:
            attributs_drop.append(var)
        else:
            continue

    if len(extra_add) == 0:
        df_copy.drop(attributs_drop, axis=1, inplace=True)

    else:
        attributs_drop = attributs_drop + extra_add
        df_copy.drop(attributs_drop, axis=1, inplace=True)

    return df_copy


#To DO
# Imputation -> Mean, Median, -1, k-nearest .. ?
# Check correlation between columns and drop redundant
# Columns with uniform distribution
# Normalize Price
