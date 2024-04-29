#imports
import pandas as pd
import numpy as np
import sklearn
## for machine learning
from sklearn import model_selection, preprocessing, feature_selection, ensemble, linear_model, metrics, decomposition
from sklearn.metrics import accuracy_score,confusion_matrix
from tensorflow.keras.models import load_model
import tensorflow as tf
## MODEL: LSTM ##
import numpy as np
from tensorflow.python.framework import ops
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from math import sqrt
from sklearn.metrics import mean_squared_error
from numpy import concatenate
import keras
from DL.utils import *

#tf.compat.v1.disable_v2_behavior()

def adaptShapesToLSTM(xtrain, xtest, ytrain, ytest, xval, yval, W, num_features):
    """
    Add an extra dimention to use in LSTM networks
    """
    train_X = np.array(xtrain)
    test_X = np.array(xtest)
    train_y = np.array(ytrain)
    test_y = np.array(ytest)
    val_X = np.array(xval)
    val_y = np.array(yval)

    train_X = train_X.reshape((train_X.shape[0], W, num_features))
    test_X = test_X.reshape((test_X.shape[0], W, num_features))
    val_X = val_X.reshape((val_X.shape[0], W, num_features))

    return train_X, train_y, test_X, test_y, val_X, val_y


def fit_lstm_model(xTrain, yTrain, xTest, yTest, xVal, yVal, scaler, W, H, num_features):

    #n_layers = 4
    #d_model = [64, 128, 64]
    los = 'mae'
    #dropout = 0
    learning_rate = 0.01
    batch = 1024
    epochs = 1
    batch = 1024

    modelo = Sequential()
    modelo.add(LSTM(50, input_shape=(xTrain.shape[1], xTrain.shape[2]), return_sequences=True))
    modelo.add(LSTM(50, input_shape=(xTrain.shape[1], xTrain.shape[2]), return_sequences=False))
    modelo.add(Dense(H))

    # optimizar y compilar
    opt = keras.optimizers.Adam(learning_rate=learning_rate)
    modelo.compile(loss=los, optimizer=opt)

    modelo.fit(x=xTrain, y=yTrain, epochs=epochs, batch_size=batch, verbose=1, validation_data=(xVal, yVal))
    modelo.summary()

    return modelo

def une(x, y, pred):
    reales = np.concatenate((x, y), axis=1)
    pred = np.concatenate((x, pred), axis=1)

    return reales, pred

def evalua_LSTM(modelo, xTest, yTest, scaler, H):
    prediction = modelo.predict(xTest)
    x_r = xTest.reshape(len(xTest), xTest.shape[1] * xTest.shape[2])
    p, r, p_df, r_df = predice(prediction, x_r, yTest, scaler, H)

    mae, mse, wape = metricas((p), (r))
    return p_df
