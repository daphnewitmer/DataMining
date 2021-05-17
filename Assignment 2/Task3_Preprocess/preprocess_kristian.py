import pandas as pd
import scipy.stats as ss

data = pd.read_csv(r'C:\Users\Kristian\Documents\VU\Data Mining\Assignment 2\pythonProject\training_set_VU_DM.csv',
                   nrows=10000)

def factor_to_categorical(columns, no_of_groups):
    """
    Convert column with many different numeric entries to few categories
    """
    for column in columns:
        data[column] = pd.qcut(data[column].values, no_of_groups, duplicates='drop').codes + 1
    return data


factor_to_categorical(['visitor_hist_adr_usd', 'prop_review_score'], 3)


#  Returns many separate arrays. How to make this a column in a Pandas dataframe?
def rank_price_within_search_id(data):
    """
    Produce a ranking of the property prices for each search_id
    """
    unique_search_ids = set(data['srch_id'])
    ranks = []
    for ids in unique_search_ids:
        ranks.append(ss.rankdata(data.loc[data['srch_id'] == ids, 'prop_log_historical_price']))
    return ranks


rank_price_within_search_id(data)