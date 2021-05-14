import Task3_Preprocess.preprocess_daph as prep_d
import Task4_Model.model_daph as model
import Task3_Preprocess.preprocess_seby as prep_s

MODE = "test"  # or "test"

attr_to_select = ['prop_starrating', 'prop_location_score1', 'price_usd']

train = prep_s.data_input("../Assignment 2/Data/training_set_VU_DM.csv", complete=False, nrows=10000)
train = prep_d.remove_nan_values(train)
train['target'] = train.apply(prep_d.add_target_attribute, axis=1)

if MODE == 'test':
    test_full = prep_s.data_input("../Assignment 2/Data/test_set_VU_DM.csv", complete=False, nrows=10000)
    test = prep_d.remove_nan_values(test_full)
if MODE == 'train':
    test = False

X_train, X_test, y_train, y_test = prep_d.prepare_data_for_model(train, test, attr_to_select)


# # Perform Classification algorithms
# predictions = model.logistic_regression(X_train, X_test, y_train, y_test, MODE)
# predictions = model.naive_bayes(X_train, X_test, y_train, y_test, MODE)
predictions = model.decision_tree(X_train, X_test, y_train, y_test, MODE)
# predictions = model.k_neighbors(X_train, X_test, y_train, y_test, MODE)

if MODE == "test":
    model.create_output(test_full, predictions)
