# Import libraries
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import pandas as pd
import DL.utils

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

def fit_MLP(xTrain, yTrain, xTest, yTest, xVal, yVal, scaler, W, H, num_features):
    ##Data are in the specific 3D format (for LSTM and others)
    input_shape = (xTrain.shape[1], xTrain.shape[2])

    # Crea un modelo MLP para regresión
    model = models.Sequential([
        layers.Flatten(input_shape=input_shape),  # Aplana los datos de entrada
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(H)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')
    #Train
    model.fit(xTrain, yTrain, epochs=10, batch_size=32, validation_data=(xVal, yVal))

    return model

def evalua_MLP(modelo, xTest, yTest, scaler, H):
    prediction = modelo.predict(xTest)
    x_r = xTest.reshape(len(xTest), xTest.shape[1] * xTest.shape[2])
    p, r, p_df, r_df = DL.utils.predice(prediction, x_r, yTest, scaler, H)

    mae, mse, wape = DL.utils.metricas((p), (r))
    return p_df