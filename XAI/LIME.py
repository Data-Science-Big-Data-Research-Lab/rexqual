import lime
from lime import lime_tabular
from lime import explanation
from lime import lime_base

def LIME (modelo, xtest, xtrain, value_cols):
    explainer = lime_tabular.LimeTabularExplainer(xtrain)
    exp = explainer.explain_instance(xtest, modelo.predict, num_features=5)
    return exp
