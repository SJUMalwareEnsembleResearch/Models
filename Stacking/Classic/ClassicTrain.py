# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 00:32:48 2020

@author: Gavin
"""
import numpy as np 
import pandas as pd 
import xgboost as xgb
from sklearn import model_selection
from sklearn.metrics import accuracy_score
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import pickle

predictions = []

dataset = pd.read_csv('all_data2_new.csv')
X = dataset.iloc[:, 34:]
Y = dataset.iloc[:, 1]

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
Y = le.fit_transform(Y)

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20, random_state = 42)

#Random Forest
rf_file = "D:\\Repos\\SJUMalwareEnsembleResearch\\Models\\Stacking\\Classic\\RF,Ada,XG\\random_forest_model.sav"
rf_model = pickle.load(open(rf_file, 'rb'))
rf_pred = rf_model.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy_score(Y_test, rf_pred)

#XGBoost
D_train = xgb.DMatrix(X_train, label= Y_train)
D_test = xgb.DMatrix(X_test, label=Y_test)
xg_file = "D:\\Repos\\SJUMalwareEnsembleResearch\\Models\\Stacking\\Classic\\RF,Ada,XG\\xgboost_model.sav"
xg_model = pickle.load(open(xg_file, 'rb'))
pred = xg_model.predict(D_test)
xg_pred = np.asarray([np.argmax(line) for line in pred])
from sklearn.metrics import accuracy_score
accuracy_score(Y_test, xg_pred)


#Rechanging the Data
dataset = pd.read_csv('all_data2_new.csv')
X = dataset.iloc[:, 34:].values
Y = dataset.iloc[:, 1].values

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
Y = le.fit_transform(Y)

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20, random_state = 42)

#HMM
def predict(X_test, modelFile): #data is 2d np array
    pred = np.empty(0, dtype=np.int8)
    for row in X_test:
        all_models = pickle.load(open(modelFile, 'rb'))
        bestScore = -9999999999
        best_model = -1
        count = -1
        for model in all_models:
            count += 1
            try:
                score = model.score(np.reshape(row, (-1, 1)))
                if score > bestScore:
                    bestScore = score
                    best_model = count
            except:
                continue
        # global scores
        # scores[best_model] += bestScore
        pred = np.append(pred, best_model).astype(np.int32)
    return pred

#HMM Bag
hmmbag_file = 'D:\\Repos\\SJUMalwareEnsembleResearch\\Models\\Stacking\\Classic\\BaggingHMM\\finalized_model'
hmmbag_pred = np.empty(0, dtype=np.int32)
for i in range(1,6):
    modelFile = hmmbag_file + str(i) + '.sav'
    hmmbag_pred = np.append(hmmbag_pred, predict(X_test, modelFile))
    print("done")
hmmbag_pred = np.reshape(hmmbag_pred, (5, -1))
hmmbag_pred = hmmbag_pred.transpose()

#HMM Boost
hmmboost_file = 'D:\\Repos\\SJUMalwareEnsembleResearch\\Models\\Stacking\\Classic\\BoostingHMM\\finalized_model'
hmmboost_pred = np.empty(0, dtype=np.int32)
for i in range(1,6):
    modelFile = hmmboost_file + str(i) + '.sav'
    hmmboost_pred = np.append(hmmboost_pred, predict(X_test, modelFile))
    print("done")
hmmboost_pred = np.reshape(hmmboost_pred, (5, -1))
hmmboost_pred = hmmboost_pred.transpose()

all_predictions = np.empty(0, dtype=np.int8)
for i in range(1945):
    all_predictions = np.append(all_predictions, rf_pred[i])
    all_predictions = np.append(all_predictions, rf_pred[i])
    all_predictions = np.append(all_predictions, rf_pred[i])
    all_predictions = np.append(all_predictions, rf_pred[i])
    all_predictions = np.append(all_predictions, rf_pred[i])
    all_predictions = np.append(all_predictions, xg_pred[i])
    all_predictions = np.append(all_predictions, xg_pred[i])
    all_predictions = np.append(all_predictions, xg_pred[i])
    all_predictions = np.append(all_predictions, xg_pred[i])
    all_predictions = np.append(all_predictions, xg_pred[i])
    all_predictions = np.append(all_predictions, hmmbag_pred[i])
    all_predictions = np.append(all_predictions, hmmboost_pred[i])
all_predictions = np.reshape(all_predictions, (-1, 20))

def findMode(array):
    arr = [0] * 21
    bestScore = -1
    best_model = -1
    for i in range(20): #need to change
        arr[array[i]] += 1
        print(arr)
        if (arr[array[i]] > bestScore):
            bestScore = arr[array[i]]
            best_model = array[i]
    return best_model

def checkPred(array):
    Y_pred = np.empty(0, dtype=np.int8)
    for row in array:
        best_pred = findMode(row)
        Y_pred = np.append(Y_pred, best_pred)
    return Y_pred

Y_pred = checkPred(all_predictions)

pred_file = "D:\\Repos\\SJUMalwareEnsembleResearch\\Models\\Stacking\\Classic\\all_predictions5.sav"
pred_file2 = "D:\\Repos\\SJUMalwareEnsembleResearch\\Models\\Stacking\\Classic\\y_pred5.sav"

Y_pred = pickle.load(open(pred_file2, 'rb'))
pickle.dump(all_predictions, open(pred_file, 'wb'))
pickle.dump(Y_pred, open(pred_file2, 'wb'))

#Metrics

#Overall Accuracy
from sklearn.metrics import accuracy_score
accuracy_score(Y_test, Y_pred)

#Balanced Accuracy
from sklearn.metrics import balanced_accuracy_score
balanced_accuracy_score(Y_test, Y_pred)


#Precision, Recall, F1Score
from sklearn.metrics import precision_recall_fscore_support
precision_recall_fscore_support(Y_test, Y_pred, average ='weighted')

from collections import Counter
from sklearn import metrics
mapping = Counter(Y_test)
#print(Counter(y_test))
mapping = dict(sorted(mapping.items()))
#--- 259.12324500083923 seconds ---

label_map = {"0":"ADLOAD","1":"AGENT","2":"ALLAPLE_A","3":"BHO","4":"BIFROSE","5":"CEEINJECT","6":"CYCBOT_G","7":"FAKEREAN","8":"HOTBAR","9":"INJECTOR","10":"ONLINEGAMES","11":"RENOS","12":"RIMECUD_A","13":"SMALL","14":"TOGA_RFN","15":"VB","16":"VBINJECT","17":"VOBFUS", "18":"VUNDO","19":"WINWEBSEC","20":"ZBOT"  }

#print(y_test)

def write_cm(cm):
    file = open("D:\\Repos\\SJUMalwareEnsembleResearch\\Models\\cm\\classic.txt","w")
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
    fig = plt.figure(figsize=(20,10))
    ax = sns.heatmap(df_cm, annot=True,cmap='Blues', fmt='g')

    plt.yticks([0.5,1.5,2.5,3.5,4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5, 19.5, 20.5], labels,va='center')

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
 
    plt.show()
    plt.close()

plot_confusion_matrix(Y_test, Y_pred)
