import Task4_Model.model_daph as model
import Task3_Preprocess.preprocess as prep

MODE = "test"  # "train" or "test"

attr_to_select = ['prop_starrating', 'prop_location_score1', 'price_usd', 'promotion_flag', 'prop_review_score', 'prop_location_score2']

train = prep.data_input("../Assignment 2/Data/training_set_VU_DM.csv", complete=True, nrows=100)
train['likelihood_of_booking'] = train.apply(prep_d.add_target_attribute, axis=1)
train = prep.impute(train, zero=True, list_missing=['prop_review_score', 'prop_location_score2'])
train = prep.remove_nan_values(train)
train = prep.normalize(train, column="srch_id", target="price_usd", log=True)
train = prep.normalize(train, column="prop_id", target="price_usd")
train = prep.normalize(train, column="srch_id", target="prop_starrating")

if MODE == 'test':
    test_full = prep.data_input("../Assignment 2/Data/test_set_VU_DM.csv", complete=True, nrows=10000)
    test = prep.impute(test_full, zero=True, list_missing=['prop_review_score', 'prop_location_score2'])
    test = prep.remove_nan_values(test)
    test = prep.normalize(test, column="srch_id", target="price_usd", log=True)
    test = prep.normalize(test, column="prop_id", target="price_usd")
    test = prep.normalize(test, column="srch_id", target="prop_starrating")
if MODE == 'train':
    test = False

X_test_all_attr, X_train, X_test, y_train, y_test, Tqid, Vqid = prep.prepare_data_for_model(train, test, attr_to_select)

predictions = model.lambda_mart(X_train, X_test, y_train, y_test, Tqid, Vqid, MODE)
# # Perform Classification algorithms
# predictions = model.logistic_regression(X_train, X_test, y_train, y_test, MODE)
# predictions = model.naive_bayes(X_train, X_test, y_train, y_test, MODE)
# predictions = model.decision_tree(X_train, X_test, y_train, y_test, MODE)
# predictions = model.k_neighbors(X_train, X_test, y_train, y_test, MODE)


model.create_output(X_test_all_attr, y_test, predictions)
