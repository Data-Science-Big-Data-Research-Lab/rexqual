import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def w_y_hr (df):
    return df.iloc[:, 0:192]

def X_e_y(df, W, H):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    df = pd.DataFrame(scaler.fit_transform(df.values))

    x = df.iloc[:,0:W]
    y = df.iloc[:, W:W+H]
    return x, y, scaler