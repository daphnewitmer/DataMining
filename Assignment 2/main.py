import Task3_Preprocess.preprocess_daph as prep_d
import Task4_Model.model_daph as model
import Task3_Preprocess.preprocess_seby as prep_s

MODE = "train"  # "train" or "test"

attr_to_select = ['prop_starrating', 'prop_location_score1', 'price_usd', 'promotion_flag']

train = prep_s.data_input("../Assignment 2/Data/training_set_VU_DM.csv", complete=False, nrows=100)
train['likelihood_of_booking'] = train.apply(prep_d.add_target_attribute, axis=1)
# train = prep_s.impute(train, median=True)
train = prep_d.remove_nan_values(train)
# train = prep_d.normalize(train, ['price_usd'])

if MODE == 'test':
    test_full = prep_s.data_input("../Assignment 2/Data/test_set_VU_DM.csv", complete=True, nrows=10000)
    test = prep_d.remove_nan_values(test_full)
    test = prep_d.normalize(test, ['price_usd'])
if MODE == 'train':
    test = False

X_test_all_attr, X_train, X_test, y_train, y_test, Tqid, Vqid = prep_d.prepare_data_for_model(train, test, attr_to_select)

predictions = model.lambda_mart(X_train, X_test, y_train, y_test, Tqid, Vqid, MODE)
# # Perform Classification algorithms
# predictions = model.logistic_regression(X_train, X_test, y_train, y_test, MODE)
# predictions = model.naive_bayes(X_train, X_test, y_train, y_test, MODE)
# predictions = model.decision_tree(X_train, X_test, y_train, y_test, MODE)
# predictions = model.k_neighbors(X_train, X_test, y_train, y_test, MODE)


model.create_output(X_test_all_attr, y_test, predictions)
