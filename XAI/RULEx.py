import re
import numpy as np
import pandas as pd
import subprocess
import os
import time
from sklearn import preprocessing
import re
import XAI

from XAI.rules import cuenta_unos


# Executing MOQAR, the bases for RULEx XAI technique
# if you are using another XAI technique, do not pay attention to this
def coge_w(df_c, num_vars, W):
    d = pd.DataFrame()
    fin = num_vars*W
    for i in range (0, fin):
        n = df_c.columns.values[i]
        columna = df_c[n]
        d[n] = columna
    return d

def divide_horizons(df_c, num_vars, W, H, path, p):
    dir = path+"/csv/"
    os.makedirs(dir, exist_ok=True)
    df_c = pd.DataFrame(df_c)
    inicio = num_vars*W
    fin = inicio + H
    for i in range(inicio, fin):
        d = pd.DataFrame()
        d = coge_w(df_c, num_vars, W)
        nombre = df_c.columns.values[i]
        columna = df_c[nombre]
        d[nombre] = columna
        # Guardamos en un nuevo archivo
        writer = dir + p + f'{i - inicio}.csv'  # guardo en Excel
        d.to_csv(writer, index=False)

def transform_to_keel(r):

    #Creating output directory if not exist
    dir = "./javaMOQAR/datos/"
    os.makedirs(dir, exist_ok=True)

    # Path to the MOQAR JAR file
    ruta_jar = r + 'evaluateXAI_SAGRA/javaMOQAR/csvtodat.jar'
    print(ruta_jar)

    # Parámetros que deseas pasar al JAR
    parametros = ["./javaMOQAR/csv/", dir]

    # Comando para ejecutar el archivo JAR
    comando = ["java", "-jar", ruta_jar] + parametros

    # Ejecutar el comando
    try:
        subprocess.run(comando, check=True)
        print("se ha ejecutado")
    except subprocess.CalledProcessError as e:
        print(f"Error al ejecutar el archivo JAR: {e}")

def generarExperimento(r):
    #Changing directory
    os.chdir(r + '/evaluateXAI_SAGRA/javaMOQAR/')

    #Now we are in the new directory
    print("Actual directory", os.getcwd())

    ruta_jar = './generarExperimento.jar'

    # Parámetros que deseas pasar al JAR
    parametros = ["0-0,1", 'false', '30', '100', '0.001', '1']

    #Executing JAR
    comando = ["java", "-jar", ruta_jar] + parametros

    try:
        subprocess.run(comando, check=True)
        print("se han generado los experimentos")
    except subprocess.CalledProcessError as e:
        print(f"Error al ejecutar el archivo JAR: {e}")

def moqar(r):
    ruta = './experimentoGeneralIJCAI_PRED.sh'

    try:
        subprocess.run(['bash', ruta, "&"])
        print("se ha ejecutado MOQAR")
    except subprocess.CalledProcessError as e:
        print(f"Error al ejecutar el archivo JAR: {e}")

    #Waiting time for MOQAR execution
    time.sleep(550)



##Processing MOQAR rules to obtain RULEx XAI output
# Obtaining rules from .out files
def encuentra_reglas(lineas):
    indices = []
    for i in range(len(lineas) - 1):
        # esto coge las reglas
        if lineas[i][0:2] == 'IF':
            indices.append(i)
    return indices

def nombres_columnas(lineas, indice_primera_regla):
    df = pd.DataFrame()
    for x in range(2, 19):
        lista = descompone_linea(lineas[indice_primera_regla + x])
        nombre = lista[0]
        nombre1 = nombre.replace(' ', '')
        new_nombre = nombre1.replace(':', '')
        df[new_nombre] = None
    return df

def descompone_linea(linea):
    resultado = []
    l = re.split(r'(\d+)', linea)
    resultado.append(l[0])  # añadimos el texto
    # Esto es necesario cuando tenemos valores decimales
    s = ''
    for i in range(1, len(l)):
        s = s + l[i]

    s1 = s.replace(',', '.')
    s2 = s1.replace('\n', '')
    if (s2 == ''):
        s2 = 0
    resultado.append(float(s2))
    return resultado

def lee_metricas(lineas):
    indices = encuentra_reglas(lineas)
    df = nombres_columnas(lineas, 1)

    #For each rule
    for i in indices:
        regla_indice = indices.index(i)  # renglon en el que está la regla

        l = []
        for x in range(2, 19):
            linea = descompone_linea(
                lineas[i + x])
            l.append(linea[1])

        df.loc[regla_indice] = l

    return df

def filtrado(dataframe, query):
    return dataframe.query(query)

def filtra_por_indice(listado, indices):
    l = []
    for i in indices:
        l.append(listado[i])
    return l

def lee_fichero(path):
    with open(path) as file:
        nombres = file.readlines()
        file.close()
    return nombres

def escribe_nombres(indice, p1):
    l = []
    for i in range(1, indice):
        # rules are in different folders
        path = './results/' + p1 + str(
            i-1) + '/Dist_crowding_distance_assignment/Obj_SopAC,Conf,Gain_Sop_0.05_Conf_0.5/' + p1 + str(
            i-1) + 'ejec_0.out'
        l.append(path)

    return l

def crea_archivo_nombres(numero_de_h, n):
    l = escribe_nombres(numero_de_h + 1, n)
    f = open("nombres.txt", "w")
    for elemento in l:
        f.write(elemento)  # Intentar escribir
        f.write('\n')
    f.close()


def descrifra_outs(r):
    # Cambiar al directorio especificado
    os.chdir(r + '/evaluateXAI_SAGRA/javaMOQAR/')
    # Ahora estás trabajando en el nuevo directorio
    print("Directorio actual:", os.getcwd())

    path = 'nombres.txt'  # contiene los nombres de los ficheros
    nombres = lee_fichero(path)

    lista_dataframes = []
    lista_reglas_filtradas = []
    lista_reglas = []
    cantidades = []
    # query= "Conf>0.8 and SopAC>0.3 and AMPLITUD<12000"

    query = "Conf>0.1 and SopAC>0.1"  # query para seleccionar las reglas que cumplen ciertas medidas
    for linea in nombres:
        l = linea.replace('\n', '')
        with open(l) as f:
            lf = f.readlines()
            f.close()

            c = ''
            # Metricas
            df = lee_metricas(lf)
            num_reglas = len(df)
            c = str(num_reglas)
            df_filtrado = filtrado(df, query)  # filtro segun la query
            lista_dataframes.append(df_filtrado)
            writer = pd.ExcelWriter(l + '.xlsx')  # guardo en Excel

            df_filtrado.to_excel(writer, sheet_name="Metricas")
            num_reglas_filtradas = len(df_filtrado)
            c = c + " " + str(num_reglas_filtradas)

            cantidades.append(c)

            # Reglas
            indices_reglas = encuentra_reglas(lf)  # devuelve los indices de todas las reglas
            reglas = filtra_por_indice(lf, indices_reglas)  # coge todas las lineas en ese índice (las reglas)
            df_r = pd.DataFrame(list(zip(indices_reglas, reglas)), columns=['ID', 'Regla'])
            lista_reglas.append(df_r)
            indices = df_filtrado.index.tolist()  # devuelve los indices de las reglas que cumplen las condiciones
            reglas_filtradas = filtra_por_indice(reglas, indices)
            df_reglas = pd.DataFrame(list(zip(indices, reglas_filtradas)), columns=['ID', 'Regla'])
            df_reglas.to_excel(writer, sheet_name="Reglas")
            lista_reglas_filtradas.append(df_reglas)

            # Cierro el excel
            # writer.save()
            writer.close()

    with open('numero_reglas.txt', 'w') as temp_file:
        for item in cantidades:
            temp_file.write("%s\n" % item)


def postcadena(cadena, x):
    a = 0
    if x in cadena:
        a = cadena.index(x)
    return (cadena[a:])

def antecadena(cadena, x):
    a = -1
    if x in cadena:
        a = cadena.index(x)
    return (cadena[:a])

def limpia(string):
    string = string.replace('\n', '')
    string = string.replace(',', '')
    string = string.replace('[', '')
    string = string.replace(']', '')
    string = string.replace('{', '')
    string = string.replace('}', '')
    return string

def descompone_linea(linea):
    resultado = []
    l = re.split(r'(\d+)', linea)
    resultado.append(l[0])  # añadimos el texto
    # Esto es necesario cuando tenemos valores decimales
    s = ''
    for i in range(1, len(l)):
        s = s + l[i]

    s1 = s.replace(',', '.')
    s2 = s1.replace('\n', '')
    if (s2 == ''):
        s2 = 0
    resultado.append(float(s2))
    return (resultado)

def coge_intervalo(string, i1, i2):  # son elementos numericos
    intervalo = []
    primero = float(coge_elemento(string, i1))
    segundo = float(coge_elemento(string, i2))
    intervalo.append(primero)
    intervalo.append(segundo)
    return intervalo

def coge_elemento(string, i1):  # son elementos numericos
    l = string.split()
    return limpia(l[i1])

def coge_trozo(serie, a, b):
    p = []
    for i in range(len(serie)):
        if i >= a and i <= b:
            p.append(serie[i])
    return p

# Function for mapping the names of the columns to numbers
def dic_columnas(lista):
    diccionario = {}
    for i in range(len(lista)):
        diccionario[lista[i]] = i
    return diccionario

def mapea_columna(dic, c):
    i = 0
    for e in dic.keys():
        if e == c:
            i = dic[e]
    return i + 1

# Function creating an empty list
def lista_vacia(n):
    ls = []
    for i in range(n + 1):
        ls.append(0)
    return ls


# WORKING WITH QARs
# Function taking the consequent (after the word THEN)
def intervalo_consecuente(string, palabra='THEN'):
    consecuente = postcadena(string, palabra)
    intervalo = coge_intervalo(consecuente, 2, 4)
    return intervalo


# Function taking the antecedent (before the world THEN)
# it requires the list of attributes that are in the antecedent (time window w_r)
def intervalo_antecedente(string, lista_columnas, palabra='THEN'):
    antecedente = antecadena(string, palabra)
    lista = []
    a = antecedente.split()
    l = [0]
    for i in range(1, len(a)):
        if a[i] == 'AND':
            l.append(i)
    for ld in l:  # ld es el indice donde hay un IF o un AND
        intervalo = coge_intervalo(antecedente, ld + 2, ld + 4)
        columna = coge_elemento(antecedente, ld + 1)
        intervalo.append(mapea_columna(dic_columnas(lista_columnas), columna))
        lista.append(intervalo)

    return lista

# From a list of rules it takes all the antecedents
def todos_los_ates(lista_reglas, lista_wr):
    antecedentes = []
    for regla in (lista_reglas):
        antecedente = intervalo_antecedente(regla, lista_wr)  # calculo el antecedente de una regla
        for e in antecedente:
            antecedentes.append([e[0], e[1], e[2]])
    return antecedentes

# From a list of rules it takes all the consequents
def todos_los_ctes(lista_reglas):
    consecuentes = []
    for regla in (lista_reglas):
        e = intervalo_consecuente(regla)  # calculo el consecuente de una regla
        consecuentes.append([e[0], e[1]])
    return consecuentes

# Writing rules in a txt file
def escribe_reglas(i, lista_reglas, path):
    f = open(path, 'r')
    mensaje = f.read()
    f = open(path, 'w')
    f.write(mensaje)
    f.write('\n' + 'HP' + str(i) + '\n')
    for elemento in lista_reglas:
        f.write(elemento)
    f.close()


##Creating the list with wr, hp and hr from the time series dataframe
def crea_listawr(dataframe):
    lista_wr = []
    for elemento in dataframe.columns:
        elemento = elemento.strip()
        lista_wr.append(elemento)
    return lista_wr

# Function counting the appearance of the attributes in the antecedent of QARs
def ocurrencia(antecedentes, num_columnas):
    l = []
    for elemento in antecedentes:
        l.append(elemento[2])
    ls = lista_vacia(num_columnas)
    for i in range(len(l)):
        wr = int(l[i])
        ls[wr] = ls[wr] + 1
    return ls

# Function taking only attributes that appear a certain number of times more than n
def ocurrencia_mayorque(ocurrencia, n):
    l = []
    for i in range(len(ocurrencia)):
        if ocurrencia[i] > n:
            l.append(i)
    return l

def ocurrencia_total(datos, p1, n, i):
    lista_wr = crea_listawr(datos)
    # Computing vector for every set of rules (from each prediction horizon)
    l = []

    # rules are in different folders
    a = i
    path = '/home/lab08/Escritorio/evaluateXAI_SAGRA/javaMOQAR/results/' + p1 + str(a) + '/Dist_crowding_distance_assignment/Obj_SopAC,Conf,Gain_Sop_0.05_Conf_0.5/' + p1 + str(a) + 'ejec_0.out.xlsx'
    df = pd.read_excel(path, sheet_name='Reglas')
    ants = todos_los_ates(df['Regla'], lista_wr)
    o = ocurrencia(ants, len(datos.columns))
    l.append(o[2: (len(o) - n)])
    d = pd.DataFrame(l)

    return d

def feature_importance_ranking(r, archivo, p, h, w, i):
    datos = archivo
    d = ocurrencia_total(datos, p, h, i)
    return d

def normalize_list(l):
    scaler = preprocessing.MinMaxScaler()
    l = l.reshape(-1, 1)
    normalizedlist = scaler.fit_transform(l)
    return normalizedlist

def importance_ranking(ranking):
    s = ranking.sum()
    normalizedlist = normalize_list(np.array(s))
    return normalizedlist

def ranking(values, columns, W):
    df = pd.DataFrame(columns=['col_name', 'feature_importance_vals'])
    df['col_name'] = columns[0:W]
    df['feature_importance_vals'] = values
    df = df.sort_values(by='feature_importance_vals', ascending=False)
    return df