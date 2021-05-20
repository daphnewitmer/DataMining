from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn import tree
import pandas as pd
from LambdaRankNN import LambdaRankNN

FOLDS = 10

"""
  Classification Algorithms
"""

def lambda_mart(X_train, X_test, y_train, y_test, Tqid, Vqid, MODE):

    print('Running lambdaMART')
    ranker = LambdaRankNN(input_size=X_train.shape[1], hidden_layer_sizes=(1000, 10), activation=('relu', 'relu'),  solver='rmsprop')  # rmsprop adamax
    ranker.fit(X_train.values, y_train.values, Tqid, epochs=4)
    y_pred = ranker.predict(X_test.values)

    if MODE == "train":
        ranker.evaluate(X_test.values, y_test.values, Vqid, eval_at=4)

    return y_pred

def naive_bayes(X_train, X_test, y_train, y_test, mode):
    gnb = GaussianNB().fit(X_train, y_train)
    cv = cross_val_score(gnb, X_train, y_train, cv=FOLDS, scoring='accuracy')
    y_pred = gnb.predict(X_test)
    if mode == 'train':
        print('Naive Bayes Accuracy Score', accuracy_score(y_test, y_pred), '--- Cross Validation Score: ', cv.mean())
    return y_pred


def decision_tree(X_train, X_test, y_train, y_test, mode):
    dt = tree.DecisionTreeClassifier(random_state=1).fit(X_train, y_train)
    cv = cross_val_score(dt, X_train, y_train, cv=FOLDS, scoring='accuracy')
    y_pred = dt.predict(X_test)
    if mode == 'train':
        print('Decision Tree Accuracy Score', accuracy_score(y_test, y_pred), '--- Cross Validation Score: ', cv.mean())
    return y_pred


def k_neighbors(X_train, X_test, y_train, y_test, mode):
    knn = KNeighborsClassifier().fit(X_train, y_train)
    cv = cross_val_score(knn, X_train, y_train, cv=FOLDS, scoring='accuracy')
    y_pred = knn.predict(X_test)
    if mode == 'train':
        print('KNeighbors Accuracy Score: ', accuracy_score(y_test, y_pred), '--- Cross Validation Score: ', cv.mean())
    return y_pred


def create_output(data, y_test, predictions):
    print('Creating output csv')

    data['predictions'] = predictions

    data = data.sort_values(by=['srch_id', 'predictions'], ascending=[True, False])
    data = data[['srch_id', 'prop_id']]

    data.to_csv("submission.csv", index=False)
