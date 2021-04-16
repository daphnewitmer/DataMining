import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn import preprocessing
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


class Titanic:
    def __init__(self, training=[], test=[], mode='train'):
        self.training = training
        self.test = test
        self.folds = 10
        self.mode = mode

    def run(self):

        # Prepare data
        train = self.handle_nan(self.training)
        train = self.create_fam_and_alone(train)
        # train = self.group_age(train)  # If performed then add AgeGroup to recoding
        train = self.extract_titles(train)  # If performed then add Title to recoding
        train = self.handle_cabin(train)
        train = self.cabin_fill_NaN(train)
        # self.survival_rate('CabinFilled', train)
        # self.plot_survival_rate(train, 'AgeGroup')
        train = self.recoding(train, ["Sex", "Embarked", "CabinFilled", "Title"])
        train = self.normalize_data(train)
        train = self.reduce_data(train, ["Ticket", "Name", "PassengerId", "SibSp", "Parch", "Cabin"])

        # self.create_heatmap(train)

        y = train["Survived"]
        X = train.drop("Survived", axis=1)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        if self.mode == 'test':
            test = self.handle_nan(self.test)
            test = self.create_fam_and_alone(test)
            test = self.extract_titles(test)  # If performed then add Title to recoding
            test = self.handle_cabin(test)
            test = self.cabin_fill_NaN(test)
            test = self.recoding(test, ["Sex", "Embarked", "CabinFilled", "Title"])
            test = self.normalize_data(test)
            test = self.reduce_data(test, ["Ticket", "Name", "PassengerId", "SibSp", "Parch", "Cabin"])

            X_test = test

        # Perform Classification algorithms
        lr = self.logistic_regression(X_train, X_test, y_train, y_test)
        nb = self.naive_bayes(X_train, X_test, y_train, y_test)
        dt = self.decisionTree(X_train, X_test, y_train, y_test)
        knn = self.KNeighbors(X_train, X_test, y_train, y_test)

        if self.mode == 'test':
            self.convert_to_csv(lr)
            print('csv created')

    def normalize_data(self, data):
        scale = StandardScaler():
        data['Age'] = scale.fit_transform(data[['Age']])
        data['SibSp'] = scale.fit_transform(data[['SibSp']])
        data['Parch'] = scale.fit_transform(data[['Parch']])
        data['Fare'] = scale.fit_transform(data[['Fare']])
        data['FamilySize'] = scale.fit_transform(data[['FamilySize']])
        data['Title'] = scale.fit_transform(data[['Title']])
        data['Embarked'] = scale.fit_transform(data[['Embarked']])
        data['Pclass'] = scale.fit_transform(data[['Pclass']])
        data['CabinFilled'] = scale.fit_transform(data[['CabinFilled']])

        return data

    def plot_survival_rate(self, data, attr):
        sns.countplot(x=sorted(data[attr]), hue=attr, data=data)
        plt.show()

    def create_heatmap(self, data):
        plt.title('Titanic - Correlations', fontsize=14)
        sns.heatmap(data.corr(), annot=True, cmap = "gray_r")
        plt.show()

    """ 
    Classification Algorithms
    """
    def logistic_regression(self, X_train, X_test, y_train, y_test):
        clf = LogisticRegression(random_state=0, max_iter=1000).fit(X_train, y_train)
        cv = cross_val_score(clf, X_train, y_train, cv=self.folds, scoring='accuracy')
        y_pred = clf.predict(X_test)

        if self.mode == 'train':
            print('Logistic Regression Accuracy Score: ', accuracy_score(y_test, y_pred), '--- Cross Validation Score: ', cv.mean())
        return y_pred

    def KNeighbors(self, X_train, X_test, y_train, y_test):
        knn = KNeighborsClassifier().fit(X_train, y_train)
        cv = cross_val_score(knn, X_train, y_train, cv=self.folds, scoring='accuracy')
        y_pred = knn.predict(X_test)

        if self.mode == 'train':
            print('KNeighbors Accuracy Score: ', accuracy_score(y_test, y_pred), '--- Cross Validation Score: ', cv.mean())
        return y_pred

    def naive_bayes(self, X_train, X_test, y_train, y_test):
        gnb = GaussianNB().fit(X_train, y_train)
        cv = cross_val_score(gnb, X_train, y_train, cv=self.folds, scoring='accuracy')
        y_pred = gnb.predict(X_test)
        if self.mode == 'train':
            print('Naive Bayes Accuracy Score', accuracy_score(y_test, y_pred), '--- Cross Validation Score: ', cv.mean())
        return y_pred

    def decisionTree(self, X_train, X_test, y_train, y_test):
        dt = tree.DecisionTreeClassifier(random_state=1).fit(X_train, y_train)
        cv = cross_val_score(dt, X_train, y_train, cv=self.folds, scoring='accuracy')
        y_pred = dt.predict(X_test)
        if self.mode == 'train':
            print('Decision Tree Accuracy Score', accuracy_score(y_test, y_pred), '--- Cross Validation Score: ', cv.mean())
        return y_pred

    def KNeighbors(self, X_train, X_test, y_train, y_test):
        knn = KNeighborsClassifier().fit(X_train, y_train)
        cv = cross_val_score(knn, X_train, y_train, cv=self.folds, scoring='accuracy')
        y_pred = knn.predict(X_test)
        if self.mode == 'train':
            print('KNeighbors Accuracy Score: ', accuracy_score(y_test, y_pred), '--- Cross Validation Score: ', cv.mean())
        return y_pred

    """ 
    Other
    """
    def convert_to_csv(self, predictions):
        df = pd.DataFrame({"PassengerId": self.test["PassengerId"].values,
                           "Survived": predictions
                           })
        df.to_csv("submission.csv", index=False)

    """
    Removes attributes or instances from the data
    """
    def reduce_data(self, data, cols):
        data = data.drop(cols, axis=1)
        return data

    def extract_titles(self, data):
        data['Title'] = data['Name'].str.split(',', expand=True)[1].str.split(' ', expand=True)[1]
        return data

    def create_fam_and_alone(self, data):
        familySize = []
        for i in data.iterrows():
            familySize.append(i[1]['SibSp'] + i[1]['Parch'])

        data['FamilySize'] = familySize
        return data

    """
    Removes or convert missing values
    """
    def handle_nan(self, data):
        cols = ["Fare", "Age"]
        for col in cols:
            data[col].fillna(data[col].mean(), inplace=True)
        data.Embarked.fillna("U", inplace=True)
        data.Cabin.fillna("U", inplace=True)
        return data

    def handle_cabin(self, data):
        # Strip numbers
        data['Cabin'] = [j[0] for j in data['Cabin']]
        # data.uniq = data.Cabin.value_counts()
        # data.cost = data.groupby('Cabin')['Fare'].mean()
        # print('Cabin Data and mean Fare\n', pd.concat([data.uniq, data.cost], axis=1,
        #                                               keys=['TRAIN Cabin', 'Train Fare']))
        # data.groupby('Cabin')['Fare'].mean().sort_values()
        return data

    def group_age(self, data):
        groups = []

        for i in data['Age']:
            if i < 1:
                groups.append("Infant")
            elif i >= 1 and i < 13:
                groups.append("Child")
            elif i >= 13 and i < 19:
                groups.append("Teenager")
            elif i >= 19 and i < 35:
                groups.append("Young Adult")
            elif i >= 35 and i < 65:
                groups.append("Adult")
            else:
                groups.append("Elderly")

        data['AgeGroup'] = groups

        return data

    def cabin_fill_NaN(self, data):
        c = 0
        cabin = []
        for i in data['Fare']:
            if i < 16:
                j = "G"
            elif i >= 16 and i < 27:
                j = "F"
            elif i >= 27 and i < 37:
                j = "T"
            elif i >= 37 and i < 43:
                j = "A"
            elif i >= 43 and i < 51:
                j = "E"
            elif i >= 51 and i < 79:
                j = "D"
            elif i >= 79 and i < 107:
                j = "C"
            else:
                j = "B"

            if data['Cabin'][c] == 'U':
                cabin.append(j)
            else:
                cabin.append(data['Cabin'][c])
            c += 1

        data['CabinFilled'] = cabin

        return data

    def cabin_fill_NaN_test(self, data):
        c = 0
        cabin = []
        for i in data['Fare']:
            if i < 17:
                j = "G"
            elif i >= 17 and i < 30:
                j = "F"
            elif i >= 30 and i < 43:
                j = "D"
            elif i >= 43 and i < 64:
                j = "A"
            elif i >= 64 and i < 103:
                j = "E"
            elif i >= 103 and i < 133:
                j = "C"
            else:
                j = "B"

            if data['Cabin'][c] == 'U':
                cabin.append(j)
            else:
                cabin.append(data['Cabin'][c])
            c += 1

        data['CabinFilled'] = cabin

        return data

    """
    Convert strings to numbers
    """
    def recoding(self, data, cols):
        le = preprocessing.LabelEncoder()
        for col in cols:
            data[col] = le.fit_transform(data[col])
        return data

    def survival_rate(self, attr, data):
        print("{:12}   ---------------------------------".format(attr.upper()))
        x = sorted(data[attr].unique())  # values in attribute
        for j in x:
            y = len(data[attr][(data[attr] == j) & (data['Survived'] == 1)])  # survived number
            z = len(data[attr][data[attr] == j])   # total number
            print('   {:<12}{:3} out of {:3} survived -  {:3.2%}'.format(j,y,z,y/z))
        return



trainingData = pd.read_csv(open('Data/train.csv'))
testData = pd.read_csv(open('Data/test.csv'))
training = Titanic(trainingData, testData, 'test')
training.run()

# --- DATA EXPLORATION --- #

"""
* age and cabin have a lot of null values
* there are integers, objects and floats
* 12 columns
"""

# trainingData.info()
# testData.info()

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
# training_num = trainingData[['Age', 'SibSp', 'Parch', 'Fare']]
# training_cat = trainingData[['Survived', 'Pclass', 'Sex', 'Ticket', 'Cabin', 'Embarked']]

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
# pivot = pd.pivot_table(trainingData, values=['Age', 'SibSp', 'Parch', 'Fare'], index='Survived')
# print(pivot['Age'])
#
# for attribute in ['Age', 'SibSp', 'Parch', 'Fare']:
#     plt.hist(pivot[attribute])
#     plt.show()



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
