#imports
import pandas as pd
import numpy as np
import sklearn


import math
import glob
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from pandas import concat
from sklearn.preprocessing import MinMaxScaler

value_cols = ['Tmed (ºC)', 'Tmax (ºC)', 'Tmin (ºC)', 'HRmed (%)',
              'HRmax (%)', 'HRmin (%)', 'RSG (kj/m2)', 'DV (graus)', 'VVmed (m/s)',
              'VVmax (m/s)', 'P (mm)', 'Tmed Relva(ºC)', 'Tmax Relva(ºC)',
              'Tmin Relva(ºC)', 'target']


def load_data(path_to_data, useNormalization=True):
    """
    Load dataset and normalize
    :param path_to_data: Path to de input dataset
    :return: Normalized dataset as one-column vector and scaler object
    """
    sagra_data = pd.DataFrame()
    for i, path in enumerate(glob.glob('./*.csv')):
        dataset = pd.read_csv(path, sep=';', decimal=',', encoding='unicode_escape',
                              na_values=[99999.900000, 9999.900000])
        dataset.columns = ['EMA', 'Data', 'Tmed (ºC)', 'Tmax (ºC)', 'Tmin (ºC)', 'HRmed (%)',
                           'HRmax (%)', 'HRmin (%)', 'RSG (kj/m2)', 'DV (graus)', 'VVmed (m/s)',
                           'VVmax (m/s)', 'P (mm)', 'Tmed Relva(ºC)', 'Tmax Relva(ºC)',
                           'Tmin Relva(ºC)', 'ET0 (mm)']
        dataset = dataset.drop('EMA', axis=1)
        dataset['name'] = path.split('\\')[1][6:-8]
        dataset = dataset.rename({'Data': 'Date'}, axis=1)
        dataset['target'] = dataset['ET0 (mm)']
        dataset['time_idx'] = list(range(len(dataset)))
        dataset['group_idx'] = i

        sagra_data = pd.concat((sagra_data, dataset))

        scaler = None
        if useNormalization:
            scaler = MinMaxScaler(feature_range=(0, 1))
            # scaler.fit_transform(sagra_data)

    return sagra_data


def missing_data_imputer(data):
    data = data.set_index(['Date', 'name'])
    data[(np.abs(stats.zscore(data)) >= 3)] = np.nan
    imputer = KNNImputer(n_neighbors=3, weights="distance")
    data.loc[:, value_cols] = imputer.fit_transform(data[value_cols].values)
    data = data.reset_index()
    data.Date = pd.to_datetime(data.Date, format=r'%d/%m/%Y %H:%M')
    return data


def prepare_data(data):
    'Function deleting non numeric columns and normalizing the others'
    data = data.drop('Date', axis=1)
    data = data.drop('name', axis=1)
    data = data.drop('ET0 (mm)', axis=1)
    data = data.drop('time_idx', axis=1)
    data = data.drop('group_idx', axis=1)
    # scaler = MinMaxScaler(feature_range=(0, 1))
    # data=scaler.fit_transform(data)
    data = pd.DataFrame(data, columns=value_cols)
    return data