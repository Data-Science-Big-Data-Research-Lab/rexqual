import pandas as pd
import random

def XAI_random(feature_list):
    feature_importance_vals = [random.random() for _ in feature_list]
    ranking = pd.DataFrame({'col_name': feature_list, 'feature_importance_vals': feature_importance_vals})
    ranking.sort_values(by=['feature_importance_vals'], ascending=False, inplace=True)

    return ranking