# Adaboost Classification

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('all_data.csv')
X = dataset.iloc[:, 33:].values
Y = dataset.iloc[:, 0].values

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
Y = le.fit_transform(Y)


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)

# Not needed for Adaboost
# Feature Scaling
# from sklearn.preprocessing import StandardScaler
# sc = StandardScaler()
# X_train = sc.fit_transform(X_train)
# X_test = sc.transform(X_test)

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
classifier = AdaBoostClassifier(n_estimators = 500, learning_rate = 1) ##Might need to change n_estimators
classifier.fit(X_train, Y_train)


print(classifier.score(X_train, Y_train))
print(classifier.score(X_test, Y_test))