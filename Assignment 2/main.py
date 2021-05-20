import Task4_Model.model_daph as model
import Task3_Preprocess.preprocess as prep

MODE = "test"  # "train" or "test"

attr_to_select = ['price_usd', 'prop_starrating', 'prop_location_score1', 'promotion_flag',
                  'prop_location_score2', 'prop_review_score', 'prop_log_historical_price']

# 'srch_children_count'
# 'srch_room_count' : correlates with adult count
# 'site_id',
# 'visitor_location_country_id': too many values should be grouped if used
# 'prop_country_id': is it the same as the search country id? srch_destination_id
# 'srch_length_of_stay'  not a big difference
# 'srch_adults_count': not a big difference
# 'srch_booking_window', 'srch_saturday_night_bool', 'promotion_flag', 'random_bool', prop_brand_bool', 'random_bool',

train = prep.data_input("../Assignment 2/Data/training_set_VU_DM.csv", complete=False, nrows=100000)
train['likelihood_of_booking'] = train.apply(prep.add_target_attribute, axis=1)
train = prep.impute(train, zero=True, list_missing=['prop_review_score', 'prop_location_score2'])
train = prep.agg_competitors(train)
train = prep.test_impute_test(train)
train = prep.remove_nan_values(train)
train = prep.normalize(train, column="srch_id", target="price_usd", log=True)
train = prep.normalize(train, column="prop_id", target="price_usd")
train = prep.normalize(train, column="srch_id", target="prop_starrating")
train = prep.normalize(train, column="srch_id", target="prop_location_score1")
train = prep.normalize(train, column="srch_id", target="prop_location_score2")
train = prep.normalize(train, column="srch_id", target="prop_review_score")
train = prep.normalize(train, column="srch_id", target="prop_log_historical_price")

if MODE == 'test':
    test_full = prep.data_input("../Assignment 2/Data/test_set_VU_DM.csv", complete=False, nrows=100000)
    test = prep.impute(test_full, zero=True, list_missing=['prop_review_score', 'prop_location_score2'])
    test = prep.agg_competitors(test)
    test = prep.test_impute_test(test)
    test = prep.remove_nan_values(test)
    test = prep.normalize(test, column="srch_id", target="price_usd", log=True)
    test = prep.normalize(test, column="prop_id", target="price_usd")
    test = prep.normalize(test, column="srch_id", target="prop_starrating")
    test = prep.normalize(test, column="srch_id", target="prop_location_score1")
    test = prep.normalize(test, column="srch_id", target="prop_location_score2")
    test = prep.normalize(test, column="srch_id", target="prop_review_score")
    test = prep.normalize(test, column="srch_id", target="prop_log_historical_price")
if MODE == 'train':
    test = False

X_test_all_attr, X_train, X_test, y_train, y_test, Tqid, Vqid = prep.prepare_data_for_model(train, test, attr_to_select)

# print('Columns in model: ' + X_train.columns)
print(X_train.head())
# exit()
predictions = model.lambda_mart(X_train, X_test, y_train, y_test, Tqid, Vqid, MODE)
# # Perform Classification algorithms
# predictions = model.logistic_regression(X_train, X_test, y_train, y_test, MODE)
# predictions = model.naive_bayes(X_train, X_test, y_train, y_test, MODE)
# predictions = model.decision_tree(X_train, X_test, y_train, y_test, MODE)
# predictions = model.k_neighbors(X_train, X_test, y_train, y_test, MODE)


model.create_output(X_test_all_attr, y_test, predictions)
