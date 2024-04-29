from sklearn.linear_model import LinearRegression
from utils import *

def LinearRegressor(X_train, y_train, X_test, y_test):
    # Crea el modelo de regresión lineal
    model = LinearRegression()
    # Entrena el modelo con los datos de entrenamiento
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    metricas(y_test, y_pred)
    return model


