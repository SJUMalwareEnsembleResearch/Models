# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 16:14:17 2020

@author: Gavin
"""
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


#Confusion Matrix #2
from sklearn.metrics import confusion_matrix
label_map = {"0":"ADLOAD","1":"AGENT","2":"ALLAPLE_A","3":"BHO","4":"BIFROSE","5":"CEEINJECT","6":"CYCBOT_G","7":"FAKEREAN","8":"HOTBAR","9":"INJECTOR","10":"ONLINEGAMES","11":"RENOS","12":"RIMECUD_A","13":"SMALL","14":"TOGA_RFN","15":"VB","16":"VBINJECT","17":"VOBFUS", "18":"VUNDO","19":"WINWEBSEC","20":"ZBOT"  }
def write_cm(cm):
    file = open("/Users/allen/Desktop/Malware-Research/confusion_matrix_files/cm_xgboost.txt","w")
    for y in range(0, 21):
        for x in range(0, 21):
            string = (str(x) + " " + str(y) + " "+ str(round(cm[x][y],4)))
            file.write(string + "\n")
    file.close()

def plot_confusion_matrix(y_true,y_predicted):
    cm = confusion_matrix(Y_test, Y_pred)
    cm = list((cm.T / cm.astype(np.float).sum(axis=1)).T)
    for y in range(0, 21):
        for x in range(0, 21):
            cm[x][y] = round(cm[x][y], 2)
    write_cm(cm)
    print ("Plotting the Confusion Matrix")
    labels = list(label_map.values())
    df_cm = pd.DataFrame(cm,index = labels,columns = labels)
    fig = plt.figure()
    res = sns.heatmap(df_cm, annot=True,cmap='Blues', fmt='g')

    plt.yticks([0.5,1.5,2.5,3.5,4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5, 19.5, 20.5, 21.5], labels,va='center')

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
 
    plt.show()
    plt.close()

plot_confusion_matrix(y_test, best_preds)

#Confusion Matrix
from sklearn.metrics import confusion_matrix
families = [ 'ADLOAD', 'AGENT' , 'ALLAPLE_A', 'BHO', 'BIFROSE', 'CEEINJECT', 'CYCBOT_G','FAKEREAN', 'HOTBAR', 'INJECTOR',
            'ONLINEGAMES', 'RENOS', 'RIMECUD_A', 'SMALL', 'TOGA_RFN', 'VB', 'VBINJECT',
            'VOBFUS', 'VUNDO', 'WINWEBSEC', 'ZBOT']
cm = confusion_matrix(Y_test, Y_pred)
df_cm = pd.DataFrame(cm, index = [i for i in families],
                  columns = [i for i in families])
plt.figure(figsize = (10,7))
sn.heatmap(df_cm, annot=True)

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

