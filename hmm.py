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

dataset = pd.read_csv('all_data2_new.csv')
X_train = dataset.iloc[1:131, 34:].values
X_test = dataset.iloc[133:134, 34:].values



mymodel = train_hmm_random_restarts(X_train)
Y = mymodel.score(X_test)
print(Y)
random_restarts = 10


def train_hmm_random_restarts(obs_seq):
	random_restarts = 50
	model = hmm.MultinomialHMM(n_components=5, n_iter=500, tol=0.5)
	model.fit(X=obs_seq)
	prev_model = model
	return prev_model
        # model.verbose=True
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




