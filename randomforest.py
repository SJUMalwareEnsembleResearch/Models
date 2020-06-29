#Random Forest Classification

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn


dataset = pd.read_csv('all_data2_new.csv')
X = dataset.iloc[:, 34:]
Y = dataset.iloc[:, 1]

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
Y = le.fit_transform(Y)

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 42)



#n_estimators = [500, 800, 1500, 2500, 5000]
n_estimators = [100]
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

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 300, min_samples_split = 2, min_samples_leaf = 2, max_features = 'sqrt', max_depth = 30)
classifier.fit(X_train, Y_train)
Y_pred = classifier.predict(X_test)



#Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)
df_cm = pd.DataFrame(cm, index = [i for i in "ABCDEFGHIJK"],
                  columns = [i for i in "ABCDEFGHIJK"])
plt.figure(figsize = (10,7))
sn.heatmap(df_cm, annot=True)

#Overall Accuracy
from sklearn.metrics import accuracy_score
accuracy_score(Y_train, classifier.predict(X_train))
accuracy_score(Y_test, Y_pred)

#Precision, Recall, F1Score
from sklearn.metrics import precision_recall_fscore_support
precision_recall_fscore_support(Y_test, Y_pred, average ='micro')
precision_recall_fscore_support(Y_test, Y_pred, average ='macro')



##RESUTS
#n estimators: 100
  #{'n_estimators': 100, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 'auto', 'max_depth': 50}
  #3 hours
#n estimators: 150
  #

##Results
# {'n_estimators': 200, 'min_samples_split': 2, 'min_samples_leaf': 2, 'max_features': 'auto', 'max_depth': 50}
#20 minutes
#0.9943703703703703 training
#0.9337777777777778 testing

##Results
#{'n_estimators': 300, 'min_samples_split': 2, 'min_samples_leaf': 2, 'max_features': 'sqrt', 'max_depth': 30}









