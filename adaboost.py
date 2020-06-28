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


#n_estimators = [500, 800, 1500, 2500, 5000]
n_estimators = [500]
min_samples_split = [2, 5, 10, 15, 20]
min_samples_leaf = [1, 2, 5, 10, 15]
max_features = ['auto', 'sqrt','log2']
#max_depth = [10, 20, 30, 40, 50, 60, 70, 80]
max_depth = [30, 40, 50, 60, 70, 80]
max_depth.append(None)




grid_param = {'n_estimators': n_estimators, 'max_features': max_features, 'max_depth':max_depth, 'min_samples_split':min_samples_split, 'min_samples_leaf': min_samples_leaf}

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
classifier = RandomForestClassifier(random_state = 0)
RFC_random = RandomizedSearchCV(estimator = classifier, param_distributions = grid_param, n_iter = 500, verbose = 2, random_state = 42, n_jobs = -1)
RFC_random.fit(X, Y)
print(RFC_random.best_score_)
print(RFC_random.best_params_)

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
classifier = AdaBoostClassifier(n_estimators = 500, learning_rate = 1) ##Might need to change n_estimators
classifier.fit(X_train, Y_train)


print(classifier.score(X_train, Y_train))
print(classifier.score(X_test, Y_test))`