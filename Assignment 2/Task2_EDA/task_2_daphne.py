import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import dataframe_image as dfi

# train = pd.read_csv("Data/training_set_VU_DM.csv")
# test = pd.read_csv("Data/test_set_VU_DM.csv")
# train = train.dropna(axis=1, how="any")
# train.to_csv('Data/train_without_null.csv')

# train = pd.read_csv("../Data/training_set_VU_DM.csv")                              # If we remove columns that contain null values, 24 of the 54 remain
test = pd.read_csv("../Data/test_set_VU_DM.csv")

data = test
print_general = False
make_plots = False
# print(train.info())                                                             # There are int, objects and floats
# print(train.nunique())

if print_general:

    """
    Explore training data in general
    """
    print('---------Explore training data in general----------')
    print('There are ' + str(len(data.index)) + ' rows in the data.')                   # Train: 4958347, Test:
    print('There are ' + str(len(data.columns)) + ' columns in the data.')
    print('    ')

    """
    Explore training data about visitors
    """
    print('---------Explore training data about visitors----------')
    print('There is data from ' + str(data.visitor_location_country_id.nunique()) + ' different countries.')        # Train: 210
    print('    ')

    """
    Explore training data about the hotels
    """
    print('---------Explore training data about the hotels----------')
    print('There is data from ' + str(data.prop_id.nunique()) + ' different hotels.')                                # Train: 129113
    print('The hotels are in ' + str(data.prop_country_id.nunique()) + ' different countries.')                      # Train: 172

    booked = data.where(data['booking_bool'] == True)
    booked = booked.dropna(subset=['booking_bool'])
    print('Hotels have been booked ' + str(booked.prop_id.count()) + ' times.')                               # Train: 138390
    print(str(booked.prop_id.nunique()) + ' different hotels have been booked.')                             # Train: 43428

    clicked = data.where(data['click_bool'] == True)
    clicked = clicked.dropna(subset=['click_bool'])
    print('Hotels have been clicked on ' + str(clicked.prop_id.count()) + ' times.')                              # Train: 221879
    print(str(clicked.prop_id.nunique()) + ' different hotels have been clicked on.')                            # Train: 57861
    print('    ')

    """
    Explore training data about search queries
    """
    print('--------- Explore training data about search queries----------')
    print('There are ' + str(data.srch_id.nunique()) + ' searches.')                               # Train: 199795
    print('Earliest search is: '+ str(data.date_time.min()))                                # Train:  1 november 2012

    print('Latest search is ' + str(data.date_time.max()))                # Train: 30 june 2013 (on kaggle is says 1st of july)
                                                                # the search queries are from a period of about 6 months

    print('People are searching for hotels in ' + str(data.srch_destination_id.nunique()) + ' different countries')       # Train:  18127
                                                         # Train: The hotels in this dataset are only from 172 countries.
    print('    ')

    """
    Explore NaN values
    """
    print('---------Explore NaN values----------')
    print('There are ' + str(len(data.dropna(axis=1, how="any").columns)) + ' without NaN values. These are:')
    print(data.dropna(axis=1, how="any").columns)
    print('There are ' + str(
        len(data.columns) - len(data.dropna(axis=1, how="any").columns)) + ' with NaN values. These are:')
    print(data.isna().columns)


numeric_attributes = data[['prop_starrating', 'prop_location_score1', 'prop_log_historical_price',
                            'price_usd', 'srch_length_of_stay', 'srch_booking_window', 'srch_adults_count',
                            'srch_children_count', 'srch_room_count']]

if make_plots:
    """
    Pivot table numeric attributes
    """
    # Make pivot table relation between numeric attributes and booking bool. It gives the mean value for booking bool 0 vs 1 for all numeric attributes
    table = pd.pivot_table(data, values=numeric_attributes, index='booking_bool')
    dfi.export(table, 'booking_bool.png')

    """
    Heatmap
    """
    # print(numeric_attributes.corr())
    sns.set(font_scale=0.7)
    sns.heatmap(numeric_attributes.corr(), annot=True, fmt=".2f", cmap="gray_r")
    plt.xticks(rotation=75)
    plt.title('Correlations between numeric attributes', fontdict={'fontsize': 12})
    plt.savefig('Heatmap-test.png', bbox_inches='tight')



# categorical_attributes = train[['prop_brand_bool', 'promotion_flag', 'prop_country_id',
#                                 'visitor_location_country_id', 'site_id', 'srch_saturday_night_bool', 'random_bool',
#                                 'click_bool', 'booking_bool']]  # 'srch_destination_id' too big for hist

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


# Make pivot table relation between numeric attributes and booking bool. It gives the mean value for booking bool 0 vs 1 for all numeric attributes
# pd.pivot_table(data, values=numeric_attributes, index='booking_bool').to_csv('Images/pivot_table_numeric_values.csv')
#
# # make pivot table for categorical attributes using srch_id to count occurences of booking_bool 0 vs 1 for all catagorical attributes
# for attribute in categorical_attributes.columns:
#     pd.pivot_table(train, index='booking_bool', columns=attribute, values='srch_id', aggfunc='count').to_csv('Images/PivotTables/'+ attribute +'.csv')
#


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
