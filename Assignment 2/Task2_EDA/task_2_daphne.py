import pandas as pd
import matplotlib.pyplot as plt

# train = pd.read_csv("Data/training_set_VU_DM.csv")
# test = pd.read_csv("Data/test_set_VU_DM.csv")
# train = train.dropna(axis=1, how="any")
# train.to_csv('Data/train_without_null.csv')

train = pd.read_csv("Data/train_without_null.csv")          # If we remove columns that contain null values, 24 of the 54 remain

"""
Explore training data in general
"""
print(train.info())                                       # There are int, objects and floats,
print(train.count)                                        # There are 4958347 rows


"""
Explore training data about visitors
"""
print(train.visitor_location_country_id.count())          # There is data from 4958347 people
print(train.visitor_location_country_id.nunique())        # These people are from 210 different countries


"""
Explore training data about the hotels
"""
print(train.prop_id.max())                                  # There is data from 140821 hotels
print(train.prop_country_id.nunique())                      # These hotels are in 172 different countries

"""
Explore training data about search queries
"""
print(train.srch_id.max())                                  # There is data from 332785 search queries
print(train.date_time.min())                                # Earliest date is 1 november 2012

print(train.date_time.max())                                # Latest date is 30 june 2013 (on kaggle is says 1st of july)
                                                            # the search queries are from a period of about 6 months
                                                            # maybe interesting to extract info about time of search query's?

print(train.srch_destination_id.nunique())                  # People are searching for hotels in 18127 different countries.
                                                            # The hotels in this dataset are only from 172 countries.

"""
Explore data about user decision (only in training data)
"""
print(train.value_counts(subset=['prop_id', 'booking_bool']))
print(train.value_counts(subset=['prop_id', 'booking_bool']).count())


"""
Explore training data about competitors
"""

"""
Plots
"""
# Creating plot of visitor country ids
# country_ids = train.visitor_location_country_id.value_counts().index.tolist()
# counts = train.visitor_location_country_id.value_counts().tolist()
# fig = plt.figure(figsize=(10, 7))
# plt.pie(counts, labels=country_ids)
# plt.title('Number of visitors per country ID')
# plt.show()

# Creating plot of hotel country ids
# country_ids = train.prop_country_id.value_counts().index.tolist()
# counts = train.prop_country_id.value_counts().tolist()
# fig = plt.figure(figsize=(10, 7))
# plt.pie(counts, labels=country_ids)
# plt.title('Number of hotels per country ID')
# plt.show()

