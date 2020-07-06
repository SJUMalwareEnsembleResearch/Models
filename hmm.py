# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 12:00:45 2020

@author: Gavin
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from hmmlearn import hmm
import tensorflow as tf
#import matplotlib.pyplot as plt
import re

dataset = pd.read_csv('/kaggle/input/all_data2_new.csv')
X_train = dataset.iloc[1:131, 34:].values
X_test = dataset.iloc[133:134, 34:].values
X_test2 = dataset.iloc[1000:1001, 34:].values
print(X_test.shape)
test = np.array([0,5,1])
print(test.shape)

mymodel = hmm.MultinomialHMM(n_components=5, n_iter=500, tol=0.5)  
mymodel.fit(X_train)

Y1 = mymodel.score(X_test)
Y2 = mymodel.score(X_test2)
print(Y)
print(Y2)

#in CSV ADLOAD is from row 1 to 162
from statistics import mean
log_prob_true = []
for i in range(131, 163):
    data = dataset.iloc[i:i+1, 34:].values
    y = mymodel.score(data)
    print(y)
    log_prob_true.append(y)


def train_hmm_random_restarts(obs_seq):
    random_restarts = 50
    model= hmm.MultinomialHMM(n_components=5, n_iter=500, tol=0.5)     # model.verbose=True
    model.fit(X=obs_seq)
    prev_model = model
#     prev_log_prob = model.monitor_.history.pop()
#     #random_restarts = 0
#     while(random_restarts!=0):
#         model= hmm.MultinomialHMM(n_components=2, n_iter=500, tol=0.5)
#         #    model.verbose=True
#         model.fit(X=obs_seq)
#         log_prob = model.monitor_.history.pop()
#         if (prev_log_prob < log_prob):
#             prev_model = model
#             prev_log_prob = log_prob
#         random_restarts -= 1
    return prev_model




