import Task4_Model.model as model
import Task3_Preprocess.preprocess as prep

MODE = "test"  # "train" or "test"

attr_to_select = ['price_usd', 'prop_starrating', 'prop_location_score1', 'promotion_flag',
                  'prop_location_score2', 'prop_review_score', 'prop_log_historical_price', 'random_bool', 'prop_brand_bool']

train = prep.data_input("../Assignment 2/Data/training_set_VU_DM.csv", complete=True, nrows=100000)
train['likelihood_of_booking'] = train.apply(prep.add_target_attribute, axis=1)
train = prep.impute(train, zero=True, list_missing=['prop_review_score', 'prop_location_score2'])
train = prep.remove_nan_values(train)
train = prep.normalize(train, column="prop_country_id", target="price_usd", log=True)  # price_usd_norm_by_prop_country_id
train = prep.normalize(train, column="srch_id", target="prop_starrating", log=True)  # prop_starrating_norm_by_srch_id
train = prep.normalize(train, column="srch_id", target="prop_location_score1", log=True)  # prop_location_score1_norm_by_srch_id
train = prep.normalize(train, column="srch_id", target="prop_location_score2", log=True)  # prop_location_score2_norm_by_srch_id
train = prep.normalize(train, column="srch_id", target="prop_review_score", log=True)  #  prop_review_score_norm_by_srch_id
train = prep.normalize(train, column="srch_id", target="prop_log_historical_price", log=True)  # prop_log_historical_price_norm_by_srch_id

if MODE == 'test':
    test_full = prep.data_input("../Assignment 2/Data/test_set_VU_DM.csv", complete=True, nrows=100000)
    test = prep.impute(test_full, zero=True, list_missing=['prop_review_score', 'prop_location_score2'])
    test = prep.remove_nan_values(test)
    test = prep.normalize(test, column="srch_id", target="price_usd", log=True)
    test = prep.normalize(test, column="srch_id", target="prop_starrating", log=True)
    test = prep.normalize(test, column="srch_id", target="prop_location_score1", log=True)
    test = prep.normalize(test, column="srch_id", target="prop_location_score2", log=True)
    test = prep.normalize(test, column="srch_id", target="prop_review_score", log=True)
    test = prep.normalize(test, column="srch_id", target="prop_log_historical_price", log=True)
if MODE == 'train':
    test = False

X_test_all_attr, X_train, X_test, y_train, y_test, Tqid, Vqid = prep.prepare_data_for_model(train, test, attr_to_select)
predictions = model.lambda_mart(X_train, X_test, y_train, y_test, Tqid, Vqid, MODE)
model.create_output(X_test_all_attr, y_test, predictions)
