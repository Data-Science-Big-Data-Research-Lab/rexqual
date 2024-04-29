
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import sklearn.ensemble

from DL.LSTM import une
from DL.utils import *


## MODEL SVM ##

def SVM (xtrain, ytrain, xtest, ytest, xval, yval, scaler_te, W, H, n):
    clf = svm.SVR(kernel='rbf')
    lp = []
    for i in range(0, H): #predice de uno en uno los horizontes de predicción
        ctrain = list(ytrain.iloc[:,i])
        ctest = list(ytest.iloc[:,i])

        clf.fit(xtrain, ctrain)
        p = clf.predict(xtest)
        lp.append(p)

    svmdf = pd.DataFrame(lp).transpose()
    p, r, p_df, r_df = predice(svmdf, xtest, ytest, scaler_te, H)
    metricas((p), (r))

    return clf, p_df

## MODEL DT ##
def decision_tree(xtrain, ytrain, xtest, ytest, xval, yval, scaler_te, W, H, n):
    dt = DecisionTreeRegressor()
    dt.fit(xtrain, ytrain)
    p = dt.predict(xtest)

    p, r, p_df, r_df= predice(p, xtest, ytest, scaler_te, H)
    mae, mse, wape = metricas((p), (r))

    return dt, p_df

def random_forest_regresor(xtrain, ytrain, xtest, ytest, xval, yval, scaler_te, W, H, n):
    rf = sklearn.ensemble.RandomForestRegressor(n_estimators=100)
    lp = []
    for i in range(0, H):  # predice de uno en uno los horizontes de predicción
        ctrain = list(ytrain.iloc[:, i])
        ctest = list(ytest.iloc[:, i])

        rf.fit(xtrain, ctrain)
        p = rf.predict(xtest)
        lp.append(p)

    svmdf = pd.DataFrame(lp).transpose()
    p, r, p_df, r_df = predice(svmdf, xtest, ytest, scaler_te, H)
    mae, mse, wape = metricas((p), (r))

    return rf, p_df