# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 00:29:17 2020

@author: Gavin
"""
import numpy as np 
import pandas as pd 
from sklearn import model_selection
from sklearn.metrics import accuracy_score
from collections import Counter
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Activation, MaxPooling1D, Dropout, Flatten, Reshape, Dense, Conv1D, LSTM,SpatialDropout1D
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
import numpy as np
import pandas as pd
import seaborn as sns
import os
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from random import randrange
from random import seed
from random import random
import pickle

#/Users/gavinwong/Desktop/Repos/SJUMalwareEnsembleResearch/Models/CNN/Boosting
def predict(row, modelFile): #data is 2d np array
    jsonFile = modelFile + '.json'
    json_file = open(jsonFile, 'r')
    loaded_model_json = json_file.read()
    json_file.close()

    loaded_model = tf.keras.models.model_from_json(loaded_model_json)
    h5File = modelFile + '.h5'
    loaded_model.load_weights(h5)
    
    y_pred = loaded_model.predict_classes(row.reshape(1, 1000))
    print(y_pred[0])
    return y_pred[0]

def checkPred(array):
    bestScore = -1
    count = -1
    best_model = -1
    for i in range(21):
        if array[i] > count:
            count = array[i]
            best_model = i
#             bestScore = scores[i]
#         elif array[i] == count and scores[i] > bestScore: --> USE IF CAN OBTAIN CONFIDENCE LEVEL
#             best_model = i
#             bestScore = scores[i]
    return best_model

df = pd.read_csv('all_data2_new.csv')
#Getting X and Y Data
X = df.iloc[:, 34:]
Y = df.iloc[:, 1]

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
Y = le.fit_transform(Y)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 23)

print(X_test.shape)
X_train = tf.reshape(X_train, (X_train.shape[0], 1000, 1))
X_test = tf.reshape(X_test, (X_test.shape[0], 1000, 1))

X_train.shape
X_test.shape

for i in range(0,1):
    file = "/Users/gavinwong/Desktop/Repos/SJUMalwareEnsembleResearch/Models/CNN/Boosting/boosted_cnn_"
    modelFile = file + str(i)
    jsonFile = modelFile + '.json'
    json_file = open(jsonFile, 'r')
    loaded_model_json = json_file.read()
    json_file.close()

    loaded_model = tf.keras.models.model_from_json(loaded_model_json)
    h5File = modelFile + '.h5'
    loaded_model.load_weights(h5File)
    
    preds = loaded_model.predict_classes(X_test)
    print(preds.shape)
    
    predFile = "/Users/gavinwong/Desktop/Repos/SJUMalwareEnsembleResearch/Models/CNN/Boosting/Test/y_pred" + str(i) + ".sav"
    pickle.dump(preds, open(predFile, 'wb'))
    
Y_pred = np.empty(0, dtype=np.int8)
for i in range(7780):
    predFile = "/Users/gavinwong/Desktop/Repos/SJUMalwareEnsembleResearch/Models/CNN/Boosting/Test/y_pred"
    array = [0] * 21
    for j in range (0,1):
        file = predFile + str(j) + ".sav"
        y_pred = pickle.load(open(file, 'rb'))
        array[y_pred[i]] += 1
    final_Pred = checkPred(array)
    Y_pred = np.append(Y_pred, final_Pred)
    print(families[final_Pred])
    array = [0] * 21
    
print(Y_pred.shape)
predBoostFile = "/Users/gavinwong/Desktop/Repos/SJUMalwareEnsembleResearch/Models/CNN/Boosting/Test/y_pred_boosted0.sav"
pickle.dump(Y_pred, open(predBoostFile, 'wb'))

from sklearn.metrics import accuracy_score
print("-------------------------")

print(accuracy_score(y_train, Y_pred))

# error = [np.empty((0,0), dtype = np.int8)] * 21
# for i in range(7284):
#     if(y_train[i] != Y_pred[i]):
#         error = np.append(error, i)
        
# errorFile = "/kaggle/working/errors0.sav"
# pickle.dump(error, open(errorFile, 'wb'))