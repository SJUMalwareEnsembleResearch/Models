#Random Forest Classification

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

dataset = pd.read_csv('all_data2_new.csv')
X = dataset.iloc[:, 34:]
Y = dataset.iloc[:, 1]

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
Y = le.fit_transform(Y)

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 42)



#n_estimators = [500, 800, 1500, 2500, 5000]
n_estimators = [300]
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
RFC_random = RandomizedSearchCV(estimator = classifier, param_distributions = grid_param, n_iter = 500, verbose = 2, random_state = 42, n_jobs = 1)
RFC_random.fit(X, Y)
print(RFC_random.best_score_)
print(RFC_random.best_params_)

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 500, min_samples_split = 2, min_samples_leaf = 1, max_features = 'auto', max_depth = 40)
classifier.fit(X_train, Y_train)
Y_pred = classifier.predict(X_test)



#Overall Accuracy
from sklearn.metrics import accuracy_score
accuracy_score(Y_train, classifier.predict(X_train))
accuracy_score(Y_test, Y_pred)

#Balanced Accuracy
from sklearn.metrics import balanced_accuracy_score
balanced_accuracy_score(Y_train, classifier.predict(X_train))
balanced_accuracy_score(Y_test, Y_pred)

#Precision, Recall, F1Score
from sklearn.metrics import precision_recall_fscore_support
precision_recall_fscore_support(Y_test, Y_pred, average ='micro')
precision_recall_fscore_support(Y_test, Y_pred, average ='macro')
precision_recall_fscore_support(Y_test, Y_pred, average ='weighted')

#Balanced Accuracy

#Confusion Matrix
from collections import Counter
from sklearn import metrics
mapping = Counter(Y_pred)
#print(Counter(y_test))
mapping = dict(sorted(mapping.items()))
#--- 259.12324500083923 seconds ---

label_map = {"0":"ADLOAD","1":"AGENT","2":"ALLAPLE_A","3":"BHO","4":"BIFROSE","5":"CEEINJECT","6":"CYCBOT_G","7":"FAKEREAN","8":"HOTBAR","9":"INJECTOR","10":"ONLINEGAMES","11":"RENOS","12":"RIMECUD_A","13":"SMALL","14":"TOGA_RFN","15":"VB","16":"VBINJECT","17":"VOBFUS", "18":"VUNDO","19":"WINWEBSEC","20":"ZBOT"  }

#print(y_test)

def write_cm(cm):
    file = open("D:\\Repos\\SJUMalwareEnsembleResearch\\Models\\cm_txt\\rf800.txt","w")
    for y in range(0, 21):
        for x in range(0, 21):
            string = (str(x) + " " + str(y) + " "+ str(round(cm[y][x],4)))
            file.write(string + "\n")
    file.close()

def plot_confusion_matrix(y_true,y_predicted):
    cm = metrics.confusion_matrix(y_true, y_predicted)
    l = list(cm)
    #print(l)
    s = 0
    for array in l:
        for value in array:
            s += value
    ooga = []
    counter = 0
    for array in l:
        array = list(array)
        array = [round(x /mapping[counter],3)  for x in array]
        ooga.append(array)
        counter += 1

    print(ooga)


    #cm = list((cm.T / cm.astype(np.float).sum(axis=1)).T)


    write_cm(ooga)
    #print ("Plotting the Confusion Matrix")


    labels = list(label_map.values())


    df_cm = pd.DataFrame(ooga,index = labels,columns = labels)
    fig = plt.figure()
    ax = sns.heatmap(df_cm, annot=True,cmap='Blues', fmt='g')

    plt.yticks([0.5,1.5,2.5,3.5,4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5, 19.5, 20.5], labels,va='center')

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
 
    plt.show()
    plt.close()

plot_confusion_matrix(Y_test, Y_pred)



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









