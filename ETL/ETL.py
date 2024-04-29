import math
import glob
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import sklearn


def all_prepared(dataset_tr, dataset_v, dataset_te, H, W, num_vars, target_position):
    # Transform data to a supervised dataset
    reframed_tr = select_target((ts_to_supervised(dataset_tr, W, H)), H, num_vars, target_position)
    reframed_v = select_target((ts_to_supervised(dataset_v, W, H)), H, num_vars, target_position)
    reframed_te = select_target((ts_to_supervised(dataset_te, W, H)), H, num_vars, target_position)

    columnas = reframed_te.columns

    xtrain, ytrain, scaler_tr = X_e_y_separator(reframed_tr, H)
    xtest, ytest, scaler_te = X_e_y_separator(reframed_te, H)
    xval, yval, scaler_v = X_e_y_separator(reframed_v, H)

    return xtrain, ytrain, xtest, ytest, xval, yval, scaler_tr, scaler_te, columnas

def X_e_y_separator(d4, H):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    d4 = pd.DataFrame(scaler.fit_transform(d4.values))

    X = d4.iloc[:, :-H]
    Y = d4[d4.columns[-H:]]
    return X, Y, scaler


def select_target(data, H, num_var, target_position):
    if target_position > num_var:
        print('Target value out of index')
    list_var = list()
    for j in range(1, num_var + 1):
        for i in range(0, H):
            if j != target_position:
                if i == 0:
                    v = 'var%d(t)' % j
                else:
                    v = ('var%d(t+%d)' % (j, i))
                list_var.append(v)

    data.drop(list_var, axis='columns', inplace=True)
    return data


def ts_to_supervised(data, W, H, dropnan=True):
    # dataframe: data (normalized)
    # W: time window
    # H: time horizon to predict
    # target_position: where the value to predict, y, is located (column index)
    n_vars = 1 if type(data) is list else data.shape[1]
    cols, names = list(), list()
    df = pd.DataFrame(data)

    # input sequence W (X)
    for i in range(W, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' %(j+1 , i)) for j in range(n_vars)]

    # prediction horizon H (y)
    for i in range(0, H):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' %(j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' %(j+1, i)) for j in range(n_vars)]

    # all together
    agg = concat(cols, axis=1)
    agg.columns = names
    if dropnan:
        agg.dropna(inplace=True)

    return agg

def df_rows(df, inicio, final):
    return df[inicio:final]

def split_data(data, train_size=0.6, test_size=0.3, val_size=0.3):
    """
    Split data into training, validation and test. Also splitted into X and Y
    :param data: Data to be splitted
    :param historical_window: Number of past samples
    :param test_size: Percentaje of the test_size
    :param val_size: Percentaje of the val_size
    :return:
    """
    train_samples = int(train_size * len(data))
    Train = df_rows(data, 0, train_samples)

    test_samples = train_samples + int(test_size * len(data))
    Test = df_rows(data, train_samples + 1, test_samples)

    val_samples = test_samples + int(val_size * len(data))
    Val = df_rows(data, test_samples + 1, val_samples)

    return Train, Test, Val


def X_e_y_regression(df, target_position):
    y = df.iloc[:, target_position]
    X = df.drop(df.columns[[target_position]], axis='columns')
    return X, y

