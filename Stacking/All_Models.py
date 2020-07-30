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

#Getting Classic Results
classic_file = "D:\\Repos\\SJUMalwareEnsembleResearch\\Models\\Stacking\\Classic\\all_predictions.sav"
classic_pred = pickle.load(open(classic_file, 'rb'))

#Getting ANN results
ann_file = "D:\\Repos\\SJUMalwareEnsembleResearch\\Models\\Stacking\\ANN\\ann_pred.sav"
ann_pred = pickle.load(open(ann_file, 'rb'))

##Testing the predictions
print(classic_pred.shape)
print(len(ann_pred))

#Helper Predict Method --> takes row and finds the prediction from both ann and classic array
def predict(row):
    arr = [0] * 21
    bestScore = -1
    best_model = -1
    classic = classic_pred[row]
    ann = ann_pred[row]
    
    #Going through the Classic Pred
    for i in range(12):
        arr[classic[i]] += 1
        if (arr[classic[i]] > bestScore):
            bestScore = arr[classic[i]]
            best_model = classic[i]
            
    #Going through the ANN Pred
    for i in range(30):
        arr[ann[i]] += 1
        if (arr[ann[i]] > bestScore):
            bestScore = arr[ann[i]]
            bets_model = ann[i]
    return best_model

Y_pred = np.empty(0, dtype=np.int32)
for i in range (1945):
    prediction = predict(i)
    Y_pred = np.append(Y_pred, prediction)

print(Y_pred.shape)
    

    


