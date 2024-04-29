import pandas as pd
import numpy as np
import XAI

def evaluate(d_k, i, mode, f):
    # To save results
    c_df = ['num_bins', 'min_sup', 'num_rules', 'global_sup', 'sopA_g', 'sopC_g', 'global_conf', 'key features', 'features in ARs', 'RExQUAL']
    mi_dataframe = pd.DataFrame(columns=c_df)

    ### Evaluate
    num_bins = 5

    min_sup = 0.1
    mi_dataframe = XAI.evaluate.evaluacion(num_bins, min_sup, mi_dataframe, d_k, i, mode, f)

    min_sup = 0.15
    mi_dataframe = XAI.evaluate.evaluacion(num_bins, min_sup, mi_dataframe, d_k, i, mode, f)

    min_sup = 0.2
    mi_dataframe = XAI.evaluate.evaluacion(num_bins, min_sup, mi_dataframe, d_k, i, mode, f)

    mi_dataframe.to_excel('metrics_table' + str(mode) + str(i) + '.xlsx')

def evaluacion(num_bins, min_sup, mi_dataframe, d_k, i, mode, k):
    bin_data, bin_columns = XAI.rules.discretice_equidist(d_k, num_bins)
    ##Apriori
    rules = XAI.rules.rules_apriori(bin_data, bin_columns, min_sup, num_bins)
    rules.to_excel('reglasAP_h' + str(i) + str(mode) + '_' + str(num_bins) + '-' + str(min_sup) + '.xlsx')
    print('Number of rules generated by Apriori: ' + str(len(rules)))

    if len(rules) != 0:
        sop, conf, sopa, sopc = XAI.rules.metrics_conj(d_k, bin_data, bin_columns, rules, num_bins)
        if num_bins == 0.15:
            q = features_rules(mode, i)
            RExQUAL = sop * conf * (q/k)

        else:
            q = 0
            RExQUAL = 0

        l = [num_bins, min_sup, len(rules), sop, sopa, sopc, conf, k, q, RExQUAL]
        mi_dataframe = pd.concat([mi_dataframe, pd.DataFrame([l])], ignore_index=True)

    return mi_dataframe


def features_rules(mode, i):
    num_bins = 5
    min_sup = 0.15

    rp = 'reglas_AP_h' + str(i) + str(mode) + '_' + str(num_bins) + '-' + str(min_sup) + '.xlsx'
    reglas = pd.read_excel(rp)
    ants = reglas['antecedents']
    todas_var = []
    for a in ants:
        # l = XAI.rules.recorre_string_sagra(todas_var, a)
        l = XAI.rules.recorre_string_elect(todas_var, a)

    combined_list = [item for sublist in todas_var for item in sublist]
    # Convertir la lista en un conjunto para eliminar duplicados
    unique_variables = set(combined_list)
    # Contar el número de variables distintas
    num_unique_variables = len(unique_variables)
    num_unique_variables

    return num_unique_variables


#Returns the time series dataframe with only the k-top features
def k_top_features(ts, ranking, k):
    df = pd.DataFrame()
    features = ranking['col_name'][0:k]
    for elemento in features:
        df[elemento] = ts.loc[:, elemento].values

    return df

def mean(lista):
    l = []
    # Filtra los elementos que son distintos de 0.0
    lista_filtrada = [elemento for elemento in lista if elemento > 0]

    for elemento in lista_filtrada:
        l.append(elemento)
    return np.sum(l)/len(l)

#Returns the time series dataframe with only the k-top features
def k_top_features(ts, ranking, k, H):
    df = pd.DataFrame()
    features = ranking['col_name'][0:k]
    for elemento in features:
        df[elemento] = ts.loc[:, elemento].values

    ultimas = ts.columns[-H:]
    hs = ts[ultimas]
    df_c = pd.concat([df, hs], axis =1)

    return df_c

def features_better_than(ts, ranking, H, i):
    df = pd.DataFrame()
    #features = select_features_mean(ranking)
    features = select_features_q(ranking)
    for element in features:
        df[element] = ts.loc[:, element].values

    index = len(ts.columns) - H + i
    ultimas = ts.columns[index]
    print(ultimas)
    hs = ts[ultimas]
    df_c = pd.concat([df, hs], axis = 1)

    return df_c, features

def select_features_q(ranking):
    l = []
    for i in range(len(ranking)):
        v = ranking['feature_importance_vals'][i]
        if (v > 0.75): #quiero alrededor del 25% de los atbts
            l.append(ranking['col_name'][i])
    return l

def features_shap(ts, ranking, H, i):
    df = pd.DataFrame()
    features = select_features_q(ranking)
    for element in features:
        df[element] = ts.loc[:, element].values

    index = len(ts.columns) - H + i
    ultimas = ts.columns[index]
    hs = ts[ultimas]
    df_c = pd.concat([df, hs], axis = 1)

    return df_c, features