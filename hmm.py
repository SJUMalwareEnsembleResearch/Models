# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 12:00:45 2020

@author: Gavin
"""

import os
from hmmlearn import hmm
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
import re

dataset = pd.read_csv('all_data2_new.csv')
X_train = dataset.iloc[1:131, 34:]
X_train = dataset.iloc[131:, 34:]

random_restarts = 10
model= hmm.MultinomialHMM(n_components=2, n_iter=500, tol=0.5)
model.fit(X=obs_seq)