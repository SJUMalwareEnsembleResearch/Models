# Adaboost Classification

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('all_data2_new.csv')
X = dataset.iloc[:, 34:].values
Y = dataset.iloc[:, 1].values

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
Y = le.fit_transform(Y)


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 42)


#n_estimators = [100, 200, 300, 500, 800, 1000]
n_estimators = [100]
learning_rate = [0.5, 1, 1.5, 2]
algorithm = ['SAMME', 'SAMME.R']




grid_param = {'n_estimators': n_estimators, 'learning_rate': learning_rate, 'algorithm' : algorithm}

from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import RandomizedSearchCV
classifier = AdaBoostClassifier()
RFC_random = RandomizedSearchCV(estimator = classifier, param_distributions = grid_param, n_iter = 500, verbose = 2, random_state = 42, n_jobs = 1)
RFC_random.fit(X, Y)
print(RFC_random.best_score_)
print(RFC_random.best_params_)  

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
classifier = AdaBoostClassifier(n_estimators = 200, learning_rate = 0.5, algorithm = 'SAMME') ##Might need to change n_estimators
classifier.fit(X_train, Y_train)


print(classifier.score(X_train, Y_train))
print(classifier.score(X_test, Y_test))