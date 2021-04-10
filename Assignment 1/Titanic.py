import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

class Titanic:
    def __init__(self, training=[], test=[]):
        self.training = training
        self.test = test

    def run(self):
        # Prepare data
        train = self.reduce_data(self.training)
        train = self.handle_nan(train)
        train = self.recoding(train)

        y = train["Survived"]
        X = train.drop("Survived", axis=1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Perform Classification algorithms
        lr = self.logistic_regression(X_train, X_test, y_train, y_test)
        nb = self.naive_bayes(X_train, X_test, y_train, y_test)
        dt = self.decisionTree(X_train, X_test, y_train, y_test)
        knn = self.KNeighbors(X_train, X_test, y_train, y_test)

        # Create output file
        # self.convert_to_csv(predictions)

    """ 
    Classification Algorithms
    """
    def logistic_regression(self, X_train, X_test, y_train, y_test):
        clf = LogisticRegression(random_state=0, max_iter=1000).fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print('Logistic Regression: ', accuracy_score(y_test, y_pred))
        return y_pred

    def naive_bayes(self, X_train, X_test, y_train, y_test):
        gnb = GaussianNB().fit(X_train, y_train)
        y_pred = gnb.predict(X_test)
        print('Naive Bayes', accuracy_score(y_test, y_pred))
        return y_pred

    def decisionTree(self, X_train, X_test, y_train, y_test):
        dt = tree.DecisionTreeClassifier(random_state=1).fit(X_train, y_train)
        y_pred = dt.predict(X_test)
        print('Decision Tree', accuracy_score(y_test, y_pred))
        return y_pred

    def KNeighbors(self, X_train, X_test, y_train, y_test):
        knn = KNeighborsClassifier().fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        print('KNeighbors: ', accuracy_score(y_test, y_pred))
        return y_pred

    """ 
    Other
    """
    def convert_to_csv(self, predictions):
        df = pd.DataFrame({"PassengerId": self.training["PassengerId"].values,
                           "Survived": predictions
                           })
        df.to_csv("submission.csv", index=False)

    """
    Removes attributes or instances from the data
    """
    def reduce_data(self, data):
        data = data.drop(["Ticket", "Cabin", "Name", "PassengerId"], axis=1)
        return data

    """
    Removes or convert missing values
    """
    def handle_nan(self, data):
        cols = ["SibSp", "Parch", "Fare", "Age"]
        for col in cols:
            data[col].fillna(data[col].median(), inplace=True)
        data.Embarked.fillna("U", inplace=True)

        return data

    """
    Convert strings to numbers
    """
    def recoding(self, data):
        le = preprocessing.LabelEncoder()
        cols = ["Sex", "Embarked"]
        for col in cols:
            data[col] = le.fit_transform(data[col])
            # test[col] = le.transform(test[col])

        return data


trainingData = pd.read_csv(open('Data/train.csv'))
testData = pd.read_csv(open('Data/test.csv'))
training = Titanic(trainingData, testData)
training.run()

# --- DATA EXPLORATION --- #

"""
* age and cabin have a lot of null values
* there are integers, objects and floats
* 12 columns
"""

# trainingData.info()

"""
* There are 891 passengers (in this training set)
* mean age is 29 sd=14 (0 - 80)
* 0 - 8 siblings / spouse
* 0 - 6 parents / children
* mean fare price 32 sd = 49 (max = 512)
"""
# print(trainingData.describe())

"""
Split into numeric and categorical 
"""
# training_num = self.training[['Age', 'SibSp', 'Parch', 'Fare']]
# training_cat = self.training[['Survived', 'Pclass', 'Sex', 'Ticket', 'Cabin', 'Embarked']]

"""
Distribution of numeric variables
"""
# for i in training_num.columns:
#     plt.hist(training_num[i])
#     plt.title(i)
#     plt.show()

"""
Correlation between numeric variables
"""
# print(training_num.corr())
# sns.heatmap(training_num.corr())
# plt.show()

"""
Compare survival rate across age, sibsp, parch and fare
Survivers seem younger, pay more, parents or children on board and less siblings/spouse on board
"""
# pivot = pd.pivot_table(training, values=['Age', 'SibSp', 'Parch', 'Fare'], index='Survived')
# print(pivot)

"""
Distribution of categorical variables
"""
# for i in training_cat.columns:
#     sns.barplot(training_cat[i].value_counts().index, training_cat[i].value_counts())
#     plt.title(i)
#     plt.show()

"""
* First class more survivers than the other classes
* Women higher change of surviving then men
* People without cabin number have higher chance to not survive, or cabin A
* Numeric tickets vs tickets with letters seem to make no difference
* Name titles 
"""

# --- DATA PREPARATION --- #

# Remove NUll values (substitute for means or make a prediction)

# Include only relevant variables

# Transforms

# --- Training Model ---#