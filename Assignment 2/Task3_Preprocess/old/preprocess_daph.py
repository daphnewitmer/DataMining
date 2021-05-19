from sklearn.preprocessing import StandardScaler
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


def drop_columns(data, attributes):
    """
    Function that drops all given attributes from the data
    :param data: pd.DataFrame
    :param attributes: array with attribute names as string
    :return: pd.DataFrame
    """
    return data.drop(attributes, axis=1)


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


def add_target_attribute_2(row):
    """
    Not working
    :param row:
    :return:
    """

    data = pd.read_csv("../Assignment 2/Data/training_set_VU_DM.csv", nrows=1000)

    prop_id_booked = data.where((data.prop_id == row.prop_id) & (data.booking_bool == True)).dropna(subset=['booking_bool', 'prop_id'])
    times_prop_booked = int(prop_id_booked.prop_id.count())

    prop_id_occurrence = data.where(data.prop_id == row.prop_id).dropna(subset=['prop_id'])
    prop_occurrence = int(prop_id_occurrence.prop_id.count())

    return float(times_prop_booked / prop_occurrence)


def normalize(data, attributes):
    """
    Function that normalizes the date
    :param data: df.DataFrame
    :param attributes: array with attributes names as string
    :return: df.DataFrame
    """

    for attr in attributes:
        scale = StandardScaler()
        data[attr] = scale.fit_transform(data[[attr]])

    return data


def normalize_2(df: pd.DataFrame, column: str, target: str, log: bool = False) -> pd.DataFrame:
    df[target] = np.log10(df[target] + 1e-5) if log else df[target]

    agg_methods = ["mean", "std"]
    df_agg = df.groupby(column).agg({target: agg_methods})
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
        Tqid = np.sort(train['srch_id'].values)
        Vqid = False
        test = test.sort_values(['srch_id'])
        train = train.sort_values(['srch_id'])
        X_train = train[attr_to_select]
        y_train = train["likelihood_of_booking"]
        y_test = False
        X_test_all_attr = test
        X_test = test[attr_to_select]
    else:
        y = train["likelihood_of_booking"]
        X = train

        X_train_all_attr, X_test_all_attr, y_train, y_test = train_test_split(X, y, test_size=0.3)

        Tqid = np.sort(X_train_all_attr['srch_id'].values)
        Vqid = np.sort(X_test_all_attr['srch_id'].values)
        X_train_all_attr = X_train_all_attr.sort_values(['srch_id'])
        X_test_all_attr = X_test_all_attr.sort_values(['srch_id'])

        X_train = X_train_all_attr[attr_to_select]
        X_test = X_test_all_attr[attr_to_select]

    return X_test_all_attr, X_train, X_test, y_train, y_test, Tqid, Vqid