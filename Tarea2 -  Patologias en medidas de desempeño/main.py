import os
import sys
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from proyecto.utils.estandarizar_df import estandarizar_df
from proyecto.utils.modelos import entrenar_modelo, predecir_modelo
from proyecto.utils.medidas_desempeno import evaluar_matriz_confusion, genera_tabla, guarda_tabla
from proyecto.utils.ajustar_clases import ajustar_clases

train_path = os.path.join(BASE_DIR, "datasets", "Train.xlsx")
test_path = os.path.join(BASE_DIR, "datasets", "Test.xlsx")

positiva = 1
nombre_algoritmos = ['1NN', '3NN', '5NN', 'SVM', 'Naive Bayes', 'RF']
df_resultados = []

if positiva == 1:
    etiquetas = [1,0]
else:
    etiquetas = [0,1]

df_train = pd.read_excel(train_path)
df_test = pd.read_excel(test_path)

df_train = ajustar_clases(df_train)
df_test = ajustar_clases(df_test)

df_train = estandarizar_df(df_train)
df_test = estandarizar_df(df_test)

X_train = df_train.drop(columns='class')
Y_train = df_train['class']

X_test = df_test.drop(columns='class')
Y_test = df_test['class']

for algoritmo in nombre_algoritmos:
    modelo = entrenar_modelo(algoritmo, X_train, Y_train)
    Y_pred = predecir_modelo(modelo, X_test)

    df_medidas = evaluar_matriz_confusion(Y_test, Y_pred, etiquetas, algoritmo, positiva)
    df_resultados.append(df_medidas)

resultados_finales = genera_tabla(df_resultados, nombre_algoritmos)
guarda_tabla(resultados_finales, positiva)