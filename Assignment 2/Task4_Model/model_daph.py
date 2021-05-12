from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn import tree
import pandas as pd

MODE = 'test'
FOLDS = 10

"""
  Classification Algorithms
"""

def naive_bayes(X_train, X_test, y_train, y_test):
    gnb = GaussianNB().fit(X_train, y_train)
    cv = cross_val_score(gnb, X_train, y_train, cv=FOLDS, scoring='accuracy')
    y_pred = gnb.predict(X_test)
    if MODE == 'train':
        print('Naive Bayes Accuracy Score', accuracy_score(y_test, y_pred), '--- Cross Validation Score: ', cv.mean())
    return y_pred


def decision_tree(X_train, X_test, y_train, y_test):
    dt = tree.DecisionTreeClassifier(random_state=1).fit(X_train, y_train)
    cv = cross_val_score(dt, X_train, y_train, cv=FOLDS, scoring='accuracy')
    y_pred = dt.predict(X_test)
    if MODE == 'train':
        print('Decision Tree Accuracy Score', accuracy_score(y_test, y_pred), '--- Cross Validation Score: ', cv.mean())
    return y_pred


def k_neighbors(X_train, X_test, y_train, y_test):
    knn = KNeighborsClassifier().fit(X_train, y_train)
    cv = cross_val_score(knn, X_train, y_train, cv=FOLDS, scoring='accuracy')
    y_pred = knn.predict(X_test)
    if MODE == 'train':
        print('KNeighbors Accuracy Score: ', accuracy_score(y_test, y_pred), '--- Cross Validation Score: ', cv.mean())
    return y_pred


def create_output(data, predictions):
    data['target'] = predictions.tolist()

    data = data.sort_values(['srch_id', 'target'])
    df = pd.DataFrame({"srch_id": data.srch_id.values,
                       "prop_id": data.prop_id
                       })
    df.to_csv("submission.csv", index=False)
    print('csv created')
