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

files = dictionary = {'ADLOAD' : 162, 'AGENT' : 184, 'ALLAPLE_A' : 986, 'BHO' : 332, 'BIFROSE' : 156, 'CEEINJECT' : 873, 'CYCBOT_G' : 597, 'FAKEREAN' : 553,
                  'HOTBAR' : 129, 'INJECTOR' : 158, 'ONLINEGAMES' : 210, 'RENOS' : 532, 'RIMECUD_A' : 153, 'SMALL' : 180, 
                  'TOGA_RFN' : 406, 'VB' : 346, 'VBINJECT' : 937, 'VOBFUS' : 929 , 'VUNDO' : 762, 'WINWEBSEC' : 837, 'ZBOT' : 303}

##ADLOAD - 80%, AGENT - 80%, AllAPLE_A : 70%, BHO: 75%, BIFROSE: 80%, CEEINJECT: 70%, CYCBOT_G: 75%, FAKEREAN: 75%, HOTBAR: 80%, INJECTOR: 80%, ONLINEGAMES: 80%, RENOS: 75%, RIMCUD_A: 80%
##SMALL: 80%, TOGA_RFN: 75%, VB: 75%, VBINJECT: 70%, VOBFUS: 70%, VUNDO: 70%, WINWEBSEC: 70%, ZBOT: 75%
dataSelect = {'ADLOAD' : (0,130), 'AGENT' : (161, 310), 'ALLAPLE_A' : (345, 1036), 'BHO' : (1331, 1582), 'BIFROSE' : (1663, 1790), 'CEEINJECT' : (1819, 2430), 'CYCBOT_G' : (2692, 3140), 'FAKEREAN' : (3290, 3705),
              'HOTBAR' : (3842, 3945), 'INJECTOR' : (3972, 4098), 'ONLINEGAMES' : (4129, 4297), 'RENOS' : (4339, 4738), 'RIMECUD_A' : (4871, 4993), 'SMALL' : (5024, 5168), 
              'TOGA_RFN' : (5204, 5508), 'VB' : (5610, 5869), 'VBINJECT' : (5956, 6612), 'VOBFUS' : (6893, 7543) , 'VUNDO' : (7822, 8355), 'WINWEBSEC' : (8584, 9170), 'ZBOT' : (9421, 9648)}

dataSelect2 = {'ADLOAD' : (0,161), 'AGENT' : (161, 345), 'ALLAPLE_A' : (345, 1331), 'BHO' : (1331, 1663), 'BIFROSE' : (1663, 1819), 'CEEINJECT' : (1819, 2692), 'CYCBOT_G' : (2692, 3290), 'FAKEREAN' : (3290, 3842),
              'HOTBAR' : (3842, 3972), 'INJECTOR' : (3972, 4129), 'ONLINEGAMES' : (4129, 4339), 'RENOS' : (4339, 4871), 'RIMECUD_A' : (4871, 5024), 'SMALL' : (5024, 5204), 
              'TOGA_RFN' : (5204, 5610), 'VB' : (5610, 5956), 'VBINJECT' : (5956, 6893), 'VOBFUS' : (6893, 7822) , 'VUNDO' : (7822, 8584), 'WINWEBSEC' : (8584, 9421), 'ZBOT' : (9421, 9724)}

families = [ 'ADLOAD', 'AGENT' , 'ALLAPLE_A', 'BHO', 'BIFROSE', 'CEEINJECT', 'CYCBOT_G','FAKEREAN', 'HOTBAR', 'INJECTOR',
            'ONLINEGAMES', 'RENOS', 'RIMECUD_A', 'SMALL', 'TOGA_RFN', 'VB', 'VBINJECT',
            'VOBFUS', 'VUNDO', 'WINWEBSEC', 'ZBOT']

from sklearn.model_selection import train_test_splitfrom sklearn.preprocessing import LabelEncoder

testX = np.empty((0,0))
testY = np.empty(0)
count = 0
# all_models = []
for i in families:
    X = dataset.iloc[dataSelect2[i][0]:dataSelect2[i][1], 34:]
    Y = dataset.iloc[dataSelect2[i][0]:dataSelect2[i][1], 1]
    le = LabelEncoder()
    Y = le.fit_transform(Y)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 42)
    count += X_test.shape[0]
    
    testX = np.append(testX, X_test).reshape(count, 1000)
    testY = np.append(testY, Y_test)
    # model = hmm.MultinomialHMM(n_components=5, n_iter=500, tol=0.5)
    # model.fit(X_train)
    # all_models.add(model)




