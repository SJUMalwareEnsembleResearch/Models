# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 00:27:25 2020

@author: Gavin
"""

import numpy as np 
import pandas as pd 
from sklearn import model_selection
from sklearn.metrics import accuracy_score
from collections import Counter
import keras
from keras.models import Sequential 
from keras.layers import Activation, MaxPooling1D, Dropout, Flatten, Reshape, Dense, Conv1D, LSTM,SpatialDropout1D
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
import numpy as np
import pandas as pd
import seaborn as sns
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import precision_score, recall_score, accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from random import randrange
from random import seed
from random import random
import pickle
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
        
df = pd.read_csv('/kaggle/input/finalopcodes/all_data2_new.csv')
print(df.shape)

def subsample(X, Y, errors, ratio=1.0):
    sampleX = np.empty((0,0), dtype = np.int8)
    sampleY = np.empty((0,0), dtype = np.int8)
    for i in errors:
        sampleX = np.append(sampleX, X.iloc[i, :].values)
        sampleY = np.append(sampleY, Y[i])
    
    n_sample = round(len(Y) * ratio)
    while len(sampleY) < n_sample:
        index = randrange(Y.shape[0])
        X_row = X.iloc[index, :].values
        Y_row = Y[index]
        sampleX = np.append(sampleX, X_row)
        sampleY = np.append(sampleY, Y_row)
    arr = [sampleX, sampleY]
    return arr

##CNN Configurations
model = Sequential()
model.add(Conv1D(filters= 64, kernel_size=3, activation ='relu',strides = 2, padding = 'valid', input_shape= (1000, 1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(filters= 128, kernel_size=3, activation ='relu',strides = 2, padding = 'valid'))
model.add(MaxPooling1D(pool_size=2))

model.add(Dropout(0.9))
model.add(MaxPooling1D(pool_size=2))

model.add(Flatten())


model.add(Dense(21)) 
model.add(Activation('softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#Getting X and Y Data
X = df.iloc[:, 34:]
Y = df.iloc[:, 1]

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
Y = le.fit_transform(Y)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 23)

print(X_test.shape)
X_test = tf.reshape(X_test, (X_test.shape[0], 1000, 1))

##Training the CNN
file = '/kaggle/working/errors0.sav' 
errors = pickle.load(open(file, 'rb'))
type(errors)

acc = []
valacc = []
file = '/kaggle/working/errors0.sav' 
errors = pickle.load(open(file, 'rb'))

print("training model", 1, "--------------------------")
sample = subsample(X_train, y_train, errors, ratio=0.6)
baggingSampleX = sample[0].reshape(-1, 1000)
baggingSampleY = sample[1]

#print(X_train.shape)
#print(X_test.shape)
print(baggingSampleX.shape)

baggingSampleX = tf.reshape(baggingSampleX, (baggingSampleX.shape[0], 1000, 1))

#print(baggingSampleX.shape)
#print(X_test.shape)

history = model.fit(baggingSampleX,baggingSampleY, epochs = 150, batch_size = 32, validation_data = (X_test, y_test), shuffle = True)


accuracy_logs = history.history["accuracy"]
val_accuracy_logs = history.history["val_accuracy"]
acc.append(accuracy_logs)
valacc.append(accuracy_logs)

file_name = "boosted_cnn_1" 

json = file_name + ".json"
h5 = file_name + ".h5"

model_json = model.to_json()
with open(json, "w") as json_file:
    json_file.write(model_json)
    model.save_weights(h5)
    
    
    