import numpy as np
import pandas as pd
import math
import glob
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from pandas import concat
import warnings
import random
import tensorflow as tf


# metricas denormalizadas
def denormaliza(scaler, x, y, prediction):
    reales = np.concatenate((x, y), axis=1)
    pred = np.concatenate((x, prediction), axis=1)

    reales_inv = scaler.inverse_transform(reales)
    pred_inv = scaler.inverse_transform(pred)

    return reales_inv, pred_inv

def predice(prediction, xtest, ytest, scaler_te, H):
    ri, pi = denormaliza(scaler_te, xtest, ytest, prediction)

    ##ahora tengo que quedarme solo con las tres ultimas columnas
    # convierto en df
    ri_df = pd.DataFrame(ri)
    pi_df = pd.DataFrame(pi)
    # slicing df.columns to access the last three columns of the dataframe
    p = np.array(pi_df[pi_df.columns[-H:]])
    r = np.array(ri_df[ri_df.columns[-H:]])

    return p, r, ri_df, pi_df

##Error metrics
def MAE(pred, true):
    return np.mean(np.abs(pred - true))

def MSE(pred, true):
    return np.mean((pred - true) ** 2)

def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))

def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / (true + EPSILON)))

EPSILON = 1e-10
def WAPE(pred, true):
    return MAE(true, pred) / (np.mean(true) + EPSILON)


def metricas(real, pred):
    mae = MAE(pred, real)
    mse = MSE(pred, real)
    wape = WAPE(pred, real)
    print('MAE: ' + str(MAE(pred, real)) + ' MSE: ' + str(MSE(pred, real)) + ' RMSE: ' + str(
        RMSE(pred, real)) + ' WAPE: ' + str(WAPE(pred, real)))
    return mae, mse, wape



