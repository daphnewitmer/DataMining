from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import gc


def remove_nan_values(data):
    """
    Drop any column with a NaN value in it
    :param data: pd.DataFrame
    :return: pd.DataFrame
    """

    print('Removing columns with NaN values')
    return data.dropna(axis=1, how="any")


def add_target_attribute(row):
    """
    Function that returns a number based on whether booking_bool and/or clicking_bool are true.
    This should reflect the likelihood of booking.
    :param row: a row from the training data (pd.DataFrame)
    :return: int
    """
    if row.booking_bool > 0:
        return int(5)
    if row.click_bool > 0:
        return int(1)
    return int(0)

def normalize(df, column, target, log=False):

    df[target] = np.log10(df[target] + 1e-5) if log else df[target]  # np.log10: Return the base 10 logarithm of the input array, element-wise.

    agg_methods = ["mean", "std"]
    df_agg = df.groupby(column).agg({target: agg_methods})  # Get mean and std per column (eg srch_id)
    df_agg.columns = df_agg.columns.droplevel()

    col = {}
    for method in agg_methods:
        col[method] = f"{target}_{method}"
    df_agg.rename(columns=col, inplace=True)

    df_norm = df.merge(df_agg.reset_index(), on=column)
    df_norm[f"{target}_norm_by_{column}"] = (df_norm[target] - df_norm[f"{target}_mean"]) / df_norm[f"{target}_std"]
    df_norm = df_norm.drop(labels=[col["mean"], col["std"]], axis=1)

    gc.collect()

    print('Finished normalizing: ' + target)
    return df_norm


def prepare_data_for_model(train, test, attr_to_select):
    """
    Splits the data into training and test parts to apply a model
    :param test: test data (df.DataFrame or boolean False)
    :param train: training data (df.DataFrame
    :param attr_to_select: array with attributes names as string
    :return: X_train, X_test, y_train, y_test (df.DataFrame)
    """

    if isinstance(test, pd.DataFrame):
        test = test.sort_values(['srch_id'])
        train = train.sort_values(['srch_id'])
        Tqid = np.sort(train['srch_id'].values)
        X_train = train[attr_to_select]
        y_train = train["likelihood_of_booking"]
        y_test = False
        X_test_all_attr = test
        X_test = test[attr_to_select]
        Vqid = False
    else:
        y = train["likelihood_of_booking"]
        X = train

        X_train_all_attr, X_test_all_attr, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42,
                                                                              shuffle=True)
        Tqid = np.sort(X_train_all_attr['srch_id'].values)
        Vqid = np.sort(X_test_all_attr['srch_id'].values)
        X_train_all_attr = X_train_all_attr.sort_values(['srch_id'])
        X_train = X_train_all_attr[attr_to_select]

        X_test_all_attr = X_test_all_attr.sort_values(['srch_id'])
        X_test = X_test_all_attr[attr_to_select]


    return X_test_all_attr, X_train, X_test, y_train, y_test, Tqid, Vqid


def data_input(path, complete=False, nrows=10000):
    """
    Function to read a datafile.
    Arguments: path, complete, nrows
    path: Filepath to the data
    complete: Read the whole dataset (True) or specific number of rows (nrows)
    nrows: Specify number of rows for input (default: 10000)
    """
    #Checks if input is training set, if yes, remove
    #"position", "gross_bookings_usd" since not in the test set
    substring = "training_set_VU_DM"

    if complete:
        df = pd.read_csv(path)
        df["date_time"] = pd.to_datetime(
            df["date_time"], format="%Y-%m-%d %H:%M:%S")

    else:
        df = pd.read_csv(path, nrows=nrows)
        df["date_time"] = pd.to_datetime(
            df["date_time"], format="%Y-%m-%d %H:%M:%S")

    return df


def impute(df, median=False, mean=False, negative=False, zero=False, list_missing=[]):
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

    if len(list_missing) == 0:
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

    elif zero:
        for i in list_missing:
            df_copy[i].fillna(0, inplace=True)
        print("Imputation with zero done")

    else:
        print("No method choosen: Missing values at: ", list_missing)

    return df_copy
