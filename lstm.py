# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 18:59:31 2020

@author: Gavin
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
print(len(tf.config.experimental.list_physical_devices('GPU')))

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.InteractiveSession(config=config)

dataset = pd.read_csv('all_data.csv')
X = dataset.iloc[:, 33:]
Y = dataset.iloc[:, 0]
X = X/30 #Check if there are 30 different opcodes
X = np.expand_dims(X, 2)
print(X.shape)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
Y = le.fit_transform(Y)

#Initializing LSTM
lstm = tf.keras.models.Sequential()

#add dropout?
#lstm.add(Dropout(0.2))

#First Layer
lstm.add(tf.keras.layers.LSTM(128, input_shape = (1000, 1), activation = 'relu', return_sequences = True))

#Second Layer
lstm.add(tf.keras.layers.LSTM(128, input_shape = (1000, 1), activation = 'relu'))

#Third Layer
lstm.add(tf.keras.layers.Dense(64, activation = 'relu'))

#Output Layer
lstm.add(tf.keras.layers.Dense(units=11, activation='softmax')) # change activation to soft-max?

# Training the lstm on the Training set and evaluating it on the Test set
lstm.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
lstm.fit(X, Y, epochs = 10, validation_split = 0.25, batch_size = 500) ##need to change