#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 14:58:58 2020

@author: gavinwong
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Activation, MaxPooling1D, Dropout, Flatten, Reshape, Dense, Conv1D, LSTM,SpatialDropout1D
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
print(tf.__version__)

dataset = pd.read_csv('all_data2_new.csv')
X = dataset.iloc[:, 34:]
Y = dataset.iloc[:, 1]
print(X.shape)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
Y = le.fit_transform(Y)

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)
X_train = tf.reshape(X_train, (X_train.shape[0], 1000, 1))
X_test = tf.reshape(X_test, (X_test.shape[0], 1000, 1))

#Initializing CNN
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
model.fit(X_train, Y_train, epochs = 150, batch_size = 32, validation_data = (X_test, Y_test), shuffle = True)


# Training the CNN on the Training set and evaluating it on the Test set
def plot_graphs(history, best):
  plt.figure(figsize=[10,4])
  # summarize history for accuracy
  plt.subplot(121)
  #plt.plot(history.history['acc'])
  plt.plot(history.history['val_acc'])
  plt.title('model accuracy across training\n best accuracy of %.02f'%best[1])
  plt.ylabel('accuracy')
  plt.xlabel('epoch')
  plt.legend(['train', 'test'], loc='upper left')

  # summarize history for loss
  plt.subplot(122)
  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.title('model loss across training\n best loss of %.02f'%best[0])
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.legend(['train', 'test'], loc='upper left')
  plt.show()