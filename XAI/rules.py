from mlxtend.frequent_patterns import apriori, fpgrowth
from mlxtend.frequent_patterns import association_rules
from mlxtend.preprocessing import TransactionEncoder
from sklearn import datasets
import pandas as pd
import math
from functools import reduce
import re

#Discretize in a certain num_bins, with the same size
def feature_dist(data, feature_name, num_bins):
    feature = data[feature_name]
    bins = pd.cut(feature, bins=num_bins, labels=[f'bin_{i+1}' for i in range(num_bins)])
    bin_columns = pd.get_dummies(bins, prefix=feature_name)
    return bin_columns

def discretice_equidist(data, num_bins):
  bin_data = pd.DataFrame()
  bin_columnas = []
  for feature_name in data.columns:
      bin_columns = feature_dist(data, feature_name, num_bins)
      bin_columnas.append(list(bin_columns.columns))
      bin_data = pd.concat([bin_data, bin_columns], axis=1)
  return bin_data, bin_columnas


def rules_apriori(bin_data, bin_columnas, min_s, H):
    # Apriori
    frequent_itemsets = apriori(bin_data, min_support=min_s, use_colnames=True)

    last_columns = bin_data.columns[-H:]

    rules = association_rules(frequent_itemsets, metric="support", min_threshold=min_s)

    rules_single_item = rules[rules['consequents'].apply(lambda x: len(x) == 1)]
    rules_filtered = rules_single_item[
        rules_single_item['consequents'].apply(lambda x: any(atributo in x for atributo in last_columns))]

    return rules_filtered


#comparar un frozenset
def compara_fs(b, nombre):
  for elemento in b:
     q = (elemento == nombre)
  return q


def recorre_string_sagra(lista, element):
    matching_strings = []
    pattern = re.compile(r'var\d+\(t-\d+\)')
    matching_strings = pattern.findall(element)
    lista.append(matching_strings)
    return lista

def recorre_string_elect(lista, element):
    matching_strings = []
    # Verificar si el elemento coincide con el patrón 'varX(t-Y)'
    # Expresión regular para encontrar el patrón deseado
    pattern = re.compile(r'frozenset\({\' (\w+)_\w+_2\'\}\)')
    # Encontrar todas las coincidencias en el string utilizando la expresión regular
    matching_strings = pattern.findall(element)
    lista.append(matching_strings)
    return lista


# Global metrics for ARs
def cumple(fila, b, bin_columnas, num_features, num_bins):
  r = False
  for i in range(num_features):
    for j in range(num_bins):
      nombre = bin_columnas[i][j]
      b1 = compara_fs(b, nombre)
      if(b1 and fila[nombre]==1):
        r = True
  return r

def soporte_una(rules, bin_data, bin_columns, num_features, num_bins):
  l = []
  for i in range(len(rules)):
    a = rules.iloc[i]['antecedents']
    c = rules.iloc[i]['consequents']

    N = len(bin_data)
    lista = [0] * N

    for j in range(len(bin_data)):
      r1 = cumple(bin_data.iloc[j], a, bin_columns, num_features, num_bins) #si cumple el antecedente
      r2 = cumple(bin_data.iloc[j], c, bin_columns, num_features, num_bins) #si cumple el consecuente

      if (r1 and r2):
        lista[j] = 1

    l.append(lista)
  return l

def cuenta_unos(lista):
  contador_unos = 0
  for elemento in lista:
    if elemento == 1:
        contador_unos += 1
  return contador_unos

def mayor_que_uno(lista):
  contador_unos = 0
  for elemento in lista:
    if elemento > 1:
        contador_unos += 1
  return contador_unos

def mayor_o_igual_que_uno(lista):
  contador_unos = 0
  for elemento in lista:
    if elemento >= 1:
        contador_unos += 1
  return contador_unos


def soporte_conjunto(data, bin_data, bin_columnas, rules, num_bins):
    num_features = len(data.columns)
    listas = soporte_una(rules, bin_data, bin_columnas, num_features, num_bins)
    union = [max(i) for i in zip(*listas)]
    soporte = cuenta_unos(union) / (len(union))

    return soporte

def cobertura_antecedente(rules, bin_data, bin_columns, num_features, num_bins):
    l = []
    for i in range(len(rules)):
        a = rules.iloc[i]['antecedents']
        N = len(bin_data)
        lista = [0] * N
        for j in range(len(bin_data)):
            r1 = cumple(bin_data.iloc[j], a, bin_columns, num_features, num_bins)  # si cumple el antecedente
            if (r1):
                lista[j] = 1
        l.append(lista)
    return l

def cobertura_consecuente(rules, bin_data, bin_columns, num_features, num_bins):
    l = []
    for i in range(len(rules)):
        a = rules.iloc[i]['consequents']
        N = len(bin_data)
        lista = [0] * N
        for j in range(len(bin_data)):
            r1 = cumple(bin_data.iloc[j], a, bin_columns, num_features, num_bins)  # si cumple el antecedente
            if (r1):
                lista[j] = 1
        l.append(lista)
    return l

def soporte_partes(data, bin_data, bin_columns, rules, num_bins):
    num_features = len(data.columns)
    ant = cobertura_antecedente(rules, bin_data, bin_columns, num_features, num_bins)
    cons = cobertura_consecuente(rules, bin_data, bin_columns, num_features, num_bins)
    ua = [max(i) for i in zip(*ant)]
    uc = [max(i) for i in zip(*cons)]
    sop_a = cuenta_unos(ua)
    sop_c = cuenta_unos(uc)
    return sop_a/len(data), sop_c/len(data)

def confianza_conjunto(data, bin_data, bin_columns, rules, num_bins, soporte):
    num_features = len(data.columns)
    listas = cobertura_antecedente(rules, bin_data, bin_columns, num_features, num_bins)
    union = [max(i) for i in zip(*listas)]
    u = cuenta_unos(union)

    return (soporte / (u / (len(data))))

def solape_global(data, bin_data, bin_columns, rules, num_bins):
    l = []
    num_features = len(data.columns)
    for i in range(len(rules)):
        a = rules.iloc[i]['antecedents']
        c = rules.iloc[i]['consequents']
        N = len(bin_data)
        lista = [0] * N
        for j in range(len(bin_data)):
            r1 = cumple(bin_data.iloc[j], a, bin_columns, num_features, num_bins)  # si cumple el antecedente
            r2 = cumple(bin_data.iloc[j], c, bin_columns, num_features, num_bins)  # si cumple el consecuente

            if (r1 and r2):
                lista[j] += 1

        l.append(lista)
    suma = [sum(x) for x in zip(*l)]
    numerador = mayor_que_uno(suma)
    denominador = mayor_o_igual_que_uno(suma)

    return numerador/denominador


def metrics_conj(data, bin_data, bin_columns, rules, num_bins):
    soporte = soporte_conjunto(data, bin_data, bin_columns, rules, num_bins)
    print('Global support: ' + str(soporte))

    confianza = confianza_conjunto(data, bin_data, bin_columns, rules, num_bins, soporte)
    print('Global conf: ' + str(confianza))

    sopa, sopc = soporte_partes(data, bin_data, bin_columns, rules, num_bins)
    print('Antecedent sup: ' + str(sopa))
    print('Consequent sup: ' + str(sopc))

    return soporte, confianza, sopa, sopc

