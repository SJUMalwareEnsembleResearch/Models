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

sess=tf.Session()

if tf.test.gpu_device_name(): 
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
   print("Please install GPU version of TF")

dataset = pd.read_csv('all_data2_new.csv')
X = dataset.iloc[:, 34:]
Y = dataset.iloc[:, 1]
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

#Third Layer
lstm.add(tf.keras.layers.Dense(64, activation = 'relu'))

#Output Layer
lstm.add(tf.keras.layers.Dense(units=24, activation='softmax')) # change activation to soft-max?

# Training the lstm on the Training set and evaluating it on the Test set
lstm.compile(optimizer = 'SGD', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
lstm.fit(X, Y, epochs = 10, validation_split = 0.25, batch_size = 250) ##need to change

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