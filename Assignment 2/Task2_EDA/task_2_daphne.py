import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# train = pd.read_csv("Data/training_set_VU_DM.csv")
# test = pd.read_csv("Data/test_set_VU_DM.csv")
# train = train.dropna(axis=1, how="any")
# train.to_csv('Data/train_without_null.csv')

train = pd.read_csv("../Data/training_set_VU_DM.csv", nrows=1000)                              # If we remove columns that contain null values, 24 of the 54 remain

"""
Explore training data in general
"""
# print(train.info())                                                             # There are int, objects and floats
# print(train.count)                                                              # There are 4958347 rows
# print(train.nunique())

numeric_attributes = train[['prop_starrating', 'prop_location_score1', 'prop_log_historical_price', 'position',
                            'price_usd', 'srch_length_of_stay', 'srch_booking_window', 'srch_adults_count',
                            'srch_children_count', 'srch_room_count']]

categorical_attributes = train[['prop_brand_bool', 'promotion_flag', 'prop_country_id',
                                'visitor_location_country_id', 'site_id', 'srch_saturday_night_bool', 'random_bool',
                                'click_bool', 'booking_bool']]  # 'srch_destination_id' too big for hist

# Plot numeric attributes
# for i in numeric_attributes.columns:
#     plt.hist(numeric_attributes[i])
#     plt.title(i)
#     plt.show()
#
# # Plot categorical attributes
# for i in categorical_attributes.columns:
#     sns.barplot(categorical_attributes[i].value_counts().index, categorical_attributes[i].value_counts())
#     plt.title(i)
#     plt.show()
#
# # Heatmap of correlations between numeric attributes
# """
# Slight correlation between srch_room_count and srch_adults_count: may contain similar information, maybe drop one of them or combine them
# Slight correlation between srch_length_of_stay and srch_booking_window
# Slight correlation between location_score1 and property_starrating
# """
# print(numeric_attributes.corr())
# sns.heatmap(numeric_attributes.corr())
# plt.show()
#
# # Make pivot table raltion between numeric attributes and booking bool. It gives the mean value for booking bool 0 vs 1 for all numeric attributes
# pd.pivot_table(train, values=numeric_attributes, index='booking_bool').to_csv(
#     'Images/pivot_table_numeric_values.csv')
#
# # make pivot table for categorical attributes using srch_id to count occurences of booking_bool 0 vs 1 for all catagorical attributes
# for attribute in categorical_attributes.columns:
#     pd.pivot_table(train, index='booking_bool', columns=attribute, values='srch_id', aggfunc='count').to_csv('Images/PivotTables/'+ attribute +'.csv')
#
#
# """
# Explore training data about visitors
# """
# print(train.visitor_location_country_id.count())          # There is data from 4958347 people
# print(train.visitor_location_country_id.nunique())        # These people are from 210 different countries
#
#
"""
Explore training data about the hotels
"""
print(train.prop_id.nunique())                                 # There is data from 129113 different hotels
# print(train.prop_country_id.nunique())                      # These hotels are in 172 different countries

booked = train.where(train['booking_bool'] == True)
booked = booked.dropna(subset=['booking_bool'])
print(booked.prop_id.count())                               # Hotels have been booked 138390 times
print(booked.prop_id.nunique())                             # 43428 different hotels are booked

clicked = train.where(train['click_bool'] == True)
clicked = clicked.dropna(subset=['click_bool'])
print(clicked.prop_id.count())                              # Hotels have been clicked on 221879 times
print(clicked.prop_id.nunique())                            # 57861 different hotels are clicked on


# """
# Explore training data about search queries
# """
# print(train.srch_id.max())                                  # There is data from 332785 search queries
# print(train.date_time.min())                                # Earliest date is 1 november 2012
#
# print(train.date_time.max())                                # Latest date is 30 june 2013 (on kaggle is says 1st of july)
#                                                             # the search queries are from a period of about 6 months
#                                                             # maybe interesting to extract info about time of search query's?
#
# print(train.srch_destination_id.nunique())                  # People are searching for hotels in 18127 different countries.
                                                            # The hotels in this dataset are only from 172 countries.

"""
Explore data about user decision (only in training data)
"""

# x = np.arange(40)
# plt.bar(x, height=counted[:40].tolist())
# plt.xticks(x, counted[:40].index.tolist(), rotation=70)
# plt.title('40 most clicked hotels')
# plt.xlabel('Hotel Ids')
# plt.ylabel('Times clicked')
# plt.show()

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
