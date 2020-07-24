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
print(len(tf.config.experimental.list_physical_devices('GPU')))

dataset = pd.read_csv('all_data2_new.csv')
X = dataset.iloc[:, 34:]
Y = dataset.iloc[:, 1]
X = X/30 #Check if there are 30 different opcodes
X = np.expand_dims(X, 2)
print(X.shape)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
Y = le.fit_transform(Y)

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)
print(X_train.shape[0])
print(X_train.shape[1])
print(X_train.shape[2])

#Initializing CNN
cnn = tf.keras.models.Sequential()

#First Layer
cnn.add(tf.keras.layers.Conv1D(filters=16, kernel_size=3, activation ='relu', input_shape= (1000, 1)))
cnn.add(tf.keras.layers.MaxPool1D(pool_size=2, strides=2))
cnn.add(tf.keras.layers.Dropout(0.2))

#Second Layer
cnn.add(tf.keras.layers.Conv1D(filters=16, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool1D(pool_size=2, strides=2))
cnn.add(tf.keras.layers.Dropout(0.2))

#Flattening - Not sure if needed
cnn.add(tf.keras.layers.Flatten())

#Full Connection
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))

#Output Layer
cnn.add(tf.keras.layers.Dense(units=24, activation='softmax')) # change activation to soft-max?

# Training the CNN on the Training set and evaluating it on the Test set
cnn.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
cnn.fit(X, Y, epochs = 1, validation_split = 0.25, batch_size = 1000) ##need to change

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