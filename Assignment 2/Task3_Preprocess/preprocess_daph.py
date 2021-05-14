from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd

def remove_nan_values(data):
    """
    Drop any column with a NaN value in it
    :param data: pd.DataFrame
    :return: pd.DataFrame
    """
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


def prepare_data_for_model(train, test, attr_to_select):
    """
    Splits the data into training and test parts to apply a model
    :param test: test data (df.DataFrame or boolean False)
    :param train: training data (df.DataFrame
    :param attr_to_select: array with attributes names as string
    :return: X_train, X_test, y_train, y_test (df.DataFrame)
    """
    y = train["target"]
    X = train[attr_to_select]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    if isinstance(test, pd.DataFrame):
        X_test = test[attr_to_select]

    return X_train, X_test, y_train, y_test