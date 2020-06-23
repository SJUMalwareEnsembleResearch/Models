#Random Forest Classification

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


dataset = pd.read_csv('')
X = dataset.iloc[:, :].values
Y = dataset.iloc[:, :].values


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)

# Not needed for Random Forest
# Feature Scaling
# from sklearn.preprocessing import StandardScaler
# sc = StandardScaler()
# X_train = sc.fit_transform(X_train)
# X_test = sc.transform(X_test)

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0) #Might need to change n_estimators
classifier.fit(X_train, Y_train)

classifier.score(X_train, Y_train)
classifier.score(X_test, Y_test)


