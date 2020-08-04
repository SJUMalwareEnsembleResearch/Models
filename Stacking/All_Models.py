# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 11:08:31 2020

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
import os

#Rechanging the Data
dataset = pd.read_csv('all_data2_new.csv')
X = dataset.iloc[:, 34:].values
Y = dataset.iloc[:, 1].values

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
Y = le.fit_transform(Y)

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20, random_state = 42)

#Getting Classic Results
classic_file = "D:\\Repos\\SJUMalwareEnsembleResearch\\Models\\Stacking\\Classic\\all_predictions5.sav"
classic_pred = pickle.load(open(classic_file, 'rb'))

#Getting ANN results
ann_file = "D:\\Repos\\SJUMalwareEnsembleResearch\\Models\\Stacking\\ANN\\ann_pred.sav"
ann_pred = pickle.load(open(ann_file, 'rb'))

##Testing the predictions
print(classic_pred.shape)
print(ann_pred[1])

#Helper Predict Method --> takes row and finds the prediction from both ann and classic array
def predict(row):
    arr = [0] * 21
    bestScore = -1
    best_model = -1
    classic = classic_pred[row]
    ann = ann_pred[row]
    
    #Going through the Classic Pred
    for i in range(20):
        arr[classic[i]] += 1
        if (arr[classic[i]] > bestScore):
            bestScore = arr[classic[i]]
            best_model = classic[i]
            
    #Going through the ANN Pred
    for i in range(30):
        arr[ann[i]] += 1
        if (arr[ann[i]] > bestScore):
            bestScore = arr[ann[i]]
            best_model = ann[i]
    return best_model

Y_pred = np.empty(0, dtype=np.int32)
for i in range (1945):
    prediction = predict(i)
    Y_pred = np.append(Y_pred, prediction)

print(Y_pred.shape)

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
    file = open("D:\\Repos\\SJUMalwareEnsembleResearch\\Models\\cm\\all_models.txt","w")
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
    

    


