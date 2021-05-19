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

    data = pd.read_csv(
        "../Assignment 2/Data/training_set_VU_DM.csv", nrows=1000)

    prop_id_booked = data.where((data.prop_id == row.prop_id) & (
        data.booking_bool == True)).dropna(subset=['booking_bool', 'prop_id'])
    times_prop_booked = int(prop_id_booked.prop_id.count())

    prop_id_occurrence = data.where(
        data.prop_id == row.prop_id).dropna(subset=['prop_id'])
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
    df_norm[f"{target}_norm_by_{column}"] = (
        df_norm[target] - df_norm[f"{target}_mean"]) / df_norm[f"{target}_std"]
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

        X_train_all_attr, X_test_all_attr, y_train, y_test = train_test_split(
            X, y, test_size=0.3)

        Tqid = np.sort(X_train_all_attr['srch_id'].values)
        Vqid = np.sort(X_test_all_attr['srch_id'].values)
        X_train_all_attr = X_train_all_attr.sort_values(['srch_id'])
        X_test_all_attr = X_test_all_attr.sort_values(['srch_id'])

        X_train = X_train_all_attr[attr_to_select]
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
        if substring in path:
            df = df.drop(["position", "gross_bookings_usd"], axis=1)

    else:
        df = pd.read_csv(path, nrows=nrows)
        df["date_time"] = pd.to_datetime(
            df["date_time"], format="%Y-%m-%d %H:%M:%S")
        if substring in path:
            df = df.drop(["position", "gross_bookings_usd"], axis=1)

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


def factor_to_categorical(columns, no_of_groups):
    """
    Convert column with many different numeric entries to few categories
    """
    for column in columns:
        data[column] = pd.qcut(data[column].values,
                               no_of_groups, duplicates='drop').codes + 1
    return data


#  Returns many separate arrays. How to make this a column in a Pandas dataframe?
def rank_price_within_search_id(data):
    """
    Produce a ranking of the property prices for each search_id
    """
    unique_search_ids = set(data['srch_id'])
    ranks = []
    for ids in unique_search_ids:
        ranks.append(ss.rankdata(
            data.loc[data['srch_id'] == ids, 'prop_log_historical_price']))
    return ranks