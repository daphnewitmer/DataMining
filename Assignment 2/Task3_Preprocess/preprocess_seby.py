import pandas as pd
import numpy as np

#To DO
# Columns with uniform distribution ??
# Interaction effects ???
# Balance click classes

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
    df: Dataframe as input
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


def correlation_drop(df, threshold):
    """
    Finds correlations between attributes
    and drops them from the dataset
    Arguments: df, threshold
    df: Dataframe as input
    threshold: Set threshold of correlation (e.g. 0.5) when columns get deleted
    """
    df_copy = df.copy()
    col_corr = set()

    corr_matrix = df_copy.corr()

    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if (corr_matrix.iloc[i, j] >= threshold) and (corr_matrix.columns[j] not in col_corr):
                colname = corr_matrix.columns[i]
                col_corr.add(colname)
                if colname in df_copy.columns:
                    del df_copy[colname]
    print(col_corr)
    return df_copy


def impute(df, median=False, mean=False, negative=False):
    """
    Imputes the the missing values either
    with median, mean or -1 (negative value)
    Arguments: df, median, mean, negative
    df: Dataframe as input
    median: Set median=True for imputation with median value of column
    mean: Set mean=True for imputation with mean value of column
    negative: Set negative=True for -1 values instead of nan's
    Attention: Always only set one argument "True"
    """
    df_copy = df.copy()

    list_missing = df_copy.columns[df_copy.isna().any()].tolist()

    if median:
        #Impute missing values with median
        for i in list_missing:
            df_copy[i].fillna(
                (df_copy[i].median()), inplace=True)
        print("Imputation with median done")

    elif mean:
        #Impute missing values with mean
        for i in list_missing:
            df_copy[i].fillna(
                (df_copy[i].mean()), inplace=True)
        print("Imputation with mean done")

    elif negative:
        for i in list_missing:
            df_copy[i].fillna(-1, inplace=True)
        print("Imputation with negative value done")

    else:
        print("No method choosen: Missing values at: ", list_missing)

    return df_copy