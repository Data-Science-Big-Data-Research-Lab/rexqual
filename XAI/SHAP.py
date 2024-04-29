from typing import List

import shap
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def SHAP (modelo, xtest, xtrain, value_cols):
    explainer = shap.Explainer(modelo.predict, xtrain)
    f = int(len(xtest) /100)
    shap_values = explainer.__call__(xtest[0:f])

    feature_names = xtrain.columns
    rf_resultX = pd.DataFrame(shap_values.values, columns=feature_names)
    vals = np.abs(rf_resultX.values).mean(0)

    shap_importance = pd.DataFrame(list(zip(feature_names, vals)),
                                   columns=['col_name', 'feature_importance_vals'])
    shap_importance.sort_values(by=['feature_importance_vals'],
                                ascending=False, inplace=True)
    return shap_importance


def DeepSHAP(model, xtest, columns, W, H):
    explainer = shap.DeepExplainer(model, xtest)
    f = int(len(xtest)/1000)
    shap_values = explainer.shap_values(xtest[0:f])

    vals = media_shap_values(shap_values, H)
    shap_importance = pd.DataFrame(list(zip(columns[0:W], vals)),
                                   columns=['col_name', 'feature_importance_vals'])
    shap_importance.sort_values(by=['feature_importance_vals'],
                                ascending=False, inplace=True)
    return shap_importance

def media_shap_values(shap_values, H):
    h = []
    if len(shap_values) == 1:
        m = [sum(map(abs, columna)) / len(columna) for columna in zip(*shap_values[0])]
        h.append(m)
    else:
        m = [sum(map(abs, columna)) / len(columna) for columna in zip(*shap_values[H])]
        h.append(m)

    h_m = [sum(columna) / len(columna) for columna in zip(*h)]
    a=[]
    for l in h_m:
        a.extend(l)
    n = normalize_list(a)
    return n

def normalize_list(lista):
    array_bidimensional = [[elemento] for elemento in lista]
    scaler = MinMaxScaler()
    normalized_array = scaler.fit_transform(array_bidimensional)
    normalized_list = [elemento[0] for elemento in normalized_array]
    return normalized_list