import Task3_Preprocess.preprocess_daph as prep
import Task4_Model.model_daph as model
import pandas as pd
from sklearn.model_selection import train_test_split

MODE = "test"

train = pd.read_csv("../Assignment 2/Data/training_set_VU_DM.csv", nrows=10000)

train = prep.remove_nan_values(train)
train['target'] = train.apply(prep.add_target_attribute, axis=1)

train = prep.drop_columns(train, ['date_time', 'prop_log_historical_price', 'position',
                                 'srch_length_of_stay', 'srch_booking_window', 'srch_adults_count',
                                 'srch_children_count', 'srch_room_count', 'prop_brand_bool', 'promotion_flag',
                                 'prop_country_id', 'visitor_location_country_id', 'site_id', 'srch_saturday_night_bool', 'random_bool',
                                 'click_bool', 'booking_bool', 'srch_destination_id', 'srch_id', 'prop_id']) # Only left in data: 'prop_starrating', 'prop_location_score1', 'price_usd'

y = train["target"]
X = train.drop("target", axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

if MODE == 'test':
    test_full = pd.read_csv("../Assignment 2/Data/test_set_VU_DM.csv", nrows=1000)
    test = prep.remove_nan_values(test_full)
    test = prep.drop_columns(test, ['date_time', 'prop_log_historical_price',
                                 'srch_length_of_stay', 'srch_booking_window', 'srch_adults_count',
                                 'srch_children_count', 'srch_room_count', 'prop_brand_bool', 'promotion_flag',
                                 'prop_country_id', 'visitor_location_country_id', 'site_id', 'srch_saturday_night_bool', 'random_bool',
                                 'srch_destination_id', 'srch_id', 'prop_id'])
    X_test = test

# # Perform Classification algorithms
# predictions = model.logistic_regression(X_train, X_test, y_train, y_test)
# predictions = model.naive_bayes(X_train, X_test, y_train, y_test)
predictions = model.decision_tree(X_train, X_test, y_train, y_test)
# predictions = model.k_neighbors(X_train, X_test, y_train, y_test)

if MODE == "test":
    model.create_output(test_full, predictions)
