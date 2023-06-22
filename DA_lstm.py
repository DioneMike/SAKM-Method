# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 16:38:57 2023

@author: michel.dione
"""

import pandas as pd
import numpy as np
from sklearn.metrics import classification_report,roc_auc_score,confusion_matrix,accuracy_score,f1_score,roc_curve
from keras.layers import Activation, Dense, Dropout, Embedding, LSTM

import re
import keras
from IPython.display import display

import os

import string

# import time

from sklearn.preprocessing import LabelEncoder

from keras.utils import pad_sequences

from keras.models import Sequential

from keras.callbacks import ReduceLROnPlateau, EarlyStopping



import random

import matplotlib.pyplot as plt

from tensorflow.keras.datasets import imdb

#Chargement des données
df = pd.read_csv('spx.csv', parse_dates=['date'], index_col='date')
fig = plt.figure()

# Représentation graphique des données
legende='indice boursier'
plt.plot(df)
plt.grid(True) 

plt.show()



# Nous allons utiliser 95 % des données et entraîner notre modèle sur celles-ci
train_size = int(len(df) * 0.95)
test_size = len(df) - train_size
train, test = df.iloc[0:train_size], df.iloc[train_size:len(df)]

# Redimensionnement des données
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler = scaler.fit(train[['close']])
train['close'] = scaler.transform(train[['close']])
test['close'] = scaler.transform(test[['close']])

# Fonction pour la division des données en sous-séquences
def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)

#Division des données

TIME_STEPS = 30
# reshape to [samples, time_steps, n_features]
X_train, y_train = create_dataset(
  train[['close']],
  train.close,
  TIME_STEPS
)
X_test, y_test = create_dataset(
  test[['close']],
  test.close,
  TIME_STEPS
)

# Auto encodeur et LSTM dans Keras
model = keras.Sequential()
model.add(keras.layers.LSTM(
    units=64,
    input_shape=(X_train.shape[1], X_train.shape[2])
))
model.add(keras.layers.Dropout(rate=0.2))
model.add(keras.layers.RepeatVector(n=X_train.shape[1]))
model.add(keras.layers.LSTM(units=64, return_sequences=True))
model.add(keras.layers.Dropout(rate=0.2))
model.add(
  keras.layers.TimeDistributed(
    keras.layers.Dense(units=X_train.shape[2])
  )
)
model.compile(loss='mae', optimizer='adam')

history_train = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.1,
    shuffle=False
)

history_test = model.fit(
    X_test, y_test,
    epochs=10,
    batch_size=32,
    validation_split=0.1,
    shuffle=False
)


# # Evaluation du modéle

loss_train = history_train.history['loss']
loss_test = history_test.history['loss']

validation_loss_train = history_train.history['val_loss']
validation_loss_test = history_test.history['val_loss']

# accuracy = history.history['accuracy']

# val_accuracy = history.history['val_accuracy']

fig = plt.gcf()

fig.set_size_inches(13,18.5)

plt.subplot(2,1,1)

plt.plot(loss_train)
plt.plot(loss_test)
# plt.plot(validation_loss)
plt.grid(True)
plt.legend(['loss_train', 'loss_test'])

plt.subplot(2,1,2)

plt.plot(validation_loss_train)

plt.plot(validation_loss_test)
plt.grid(True)
plt.legend(['validation_loss_train', 'validation_loss_test'])

plt.show()



# Trouver les anomalies :
    #Calcul de l'erreur absolue moyenne sur bas d'apprentissage
X_train_pred = model.predict(X_train)
train_mae_loss = np.mean(np.abs(X_train_pred - X_train), axis=1)
plt.hist(train_mae_loss,density=True)
plt.grid(True)
plt.show()




THRESHOLD = 0.65
X_test_pred = model.predict(X_test)
test_mae_loss = np.mean(np.abs(X_test_pred - X_test), axis=1)


test_score_df = pd.DataFrame(index=test[TIME_STEPS:].index)
test_score_df['loss'] = test_mae_loss
test_score_df['threshold'] = THRESHOLD
test_score_df['anomaly'] = test_score_df.loss > test_score_df.threshold
test_score_df['close'] = test[TIME_STEPS:].close

# plt.plot(test_score_df['threshold'])
# plt.plot(test_score_df.loss )
anomalies = test_score_df[test_score_df.anomaly == True]

plt.plot(anomalies)
