# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 18:59:31 2020

@author: Gavin
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Activation, MaxPooling1D, Dropout, Flatten, Reshape, Dense, Conv1D, LSTM, SpatialDropout1D
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.InteractiveSession(config=config)

sess=tf.Session()

if tf.test.gpu_device_name(): 
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
   print("Please install GPU version of TF")

df = pd.read_csv('all_data2_new.csv')
#Getting X and Y Data
X = df.iloc[:, 34:]
Y = df.iloc[:, 1]

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
Y = le.fit_transform(Y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 23)

print(X_test.shape)
X_train = tf.reshape(X_train, (X_train.shape[0], 1000, 1))
X_test = tf.reshape(X_test, (X_test.shape[0], 1000, 1))
print(X_train.shape)


#Initializing LSTM

model = Sequential()
model.add(LSTM(32, input_shape=(1000, 1)))
model.add(Dropout(0.2))
model.add(Dense(21,activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train, epochs = 21,batch_size=1,validation_data = (X_test, y_test), shuffle = True)


# model = Sequential()
# model.add(Conv1D(filters= 32, kernel_size=3, activation ='relu', input_shape= (1000, 1)))
# model.add(MaxPooling1D(pool_size=(2)))

# model.add(Dense(512))
# model.add(Dropout(0.8))

# model.add(Flatten())
# model.add(Dense(256))


# model.add(Dense(128))

# model.add(Activation('relu'))


# model.add(Dense(24))
# model.add(Activation('softmax'))

# from keras.optimizers import SGD
# opt = SGD(lr=0.02)

# model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

# history = model.fit(X, Y, epochs = 30, validation_split = 0.1, batch_size = 16, shuffle=False)
# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])

# plt.title('model accuracy')
# plt.ylabel('accuracy and loss')
# plt.xlabel('epoch')

# plt.legend(['acc', 'val acc' ], loc='upper left')
# plt.show()
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model accuracy and loss')
# plt.ylabel('accuracy and loss')
# plt.xlabel('epoch')

# plt.legend(['loss', 'val loss' ], loc='upper left')
# plt.show()