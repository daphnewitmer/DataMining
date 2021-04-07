import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

training = pd.read_csv(open('Data/train.csv'))

"""
* age and cabin have a lot of null values
* there are integers, objects and floats
* 12 columns
"""

# training.info()

"""
* There are 891 passengers (registered)
* 1 survived (of the registered passengers)
* mean age is 29 sd=14 (0 - 80)
* 0 - 8 siblings / spouse
* 0 - 6 parents / children
* mean fare price 32 sd = 49 (max = 512)
"""
# print(training.describe())

"""
Split into numeric and categorical 
"""
training_num = training[['Age', 'SibSp', 'Parch', 'Fare']]
training_cat = training[['Survived', 'Pclass', 'Sex', 'Ticket', 'Cabin', 'Embarked']]

"""
Distribution of numeric variables
"""
for i in training_num.columns:
    plt.hist(training_num[i])
    plt.title(i)
    plt.show()

"""
Distribution of numeric variables
"""
print(training_num.corr())
sns.heatmap(training_num.corr())

"""
Distribution of categorical variables
"""
# for i in training_cat.columns:
#     plt.hist(training_cat[i])
#     plt.title(i)
#     plt.show()
