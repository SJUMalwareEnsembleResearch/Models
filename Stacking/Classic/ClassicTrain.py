# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 00:32:48 2020

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
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from random import randrange
from random import seed
from random import random
import pickle

predictions = []

dataset = pd.read_csv('all_data2_new.csv')
X = dataset.iloc[:, 34:]
Y = dataset.iloc[:, 1]

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
Y = le.fit_transform(Y)

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 42)

#Random Forest
rf_file = "D:\\Repos\\SJUMalwareEnsembleResearch\\Models\\Stacking\\Classic\\RF, Ada, XG\\random_forest_model.sav"
rf_model = pickle.load(rf_file, 'rb')
rf_pred = rf_model.predict(X_test)

#Adaboost
ada_file = "D:\\Repos\\SJUMalwareEnsembleResearch\\Models\\Stacking\\Classic\\RF, Ada, XG\\random_forest_model.sav"
ada_model = pickle.load(ada_file, 'rb')
ad_pred = ada_model.predict(X_test)

#HMM

def predict(row, modelFile): #data is 2d np array
    all_models = pickle.load(open(modelFile, 'rb'))
    bestScore = -9999999999
    best_model = 0
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
    global scores
    scores[best_model] += bestScore
    return best_model

def checkPred(array):
    bestScore = -1
    count = -1
    best_model = -1
    for i in range(21):
        if array[i] > count:
            count = array[i]
            best_model = i
            bestScore = scores[i]
        elif array[i] == count and scores[i] > bestScore:
            best_model = i
            bestScore = scores[i]
    return best_model