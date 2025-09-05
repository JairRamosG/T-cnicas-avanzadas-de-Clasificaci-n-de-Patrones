import pandas as pd
import os

from generar_particiones import generar_particiones_estratificadas
from modelos import entrenar_modelo, predecir_modelo
from medidas_desempeño import calcula_medidas

base_path = os.path.dirname(__file__)
csv_path = os.path.join(base_path, 'datasets', 'crx_limpio.csv')
df = pd.read_csv(csv_path)

particiones = generar_particiones_estratificadas(df, 100)

medidas_1NN = []

for X_Train, Y_Train, X_Test, Y_Test in particiones:
    
    modelo = '1NN'
    one_NN = entrenar_modelo(modelo, X_Train, Y_Train)
    Y_pred_one_NN = predecir_modelo(one_NN, X_Test)
    df_medidas = calcula_medidas(Y_Test, Y_pred_one_NN)
    medidas_1NN.append(df_medidas)

medidas_1NN = pd.concat(medidas_1NN).reset_index(drop = True)
medidas_1NN['Iteración'] = medidas_1NN.groupby('Medida').cumcount() + 1
resultado_final = medidas_1NN.pivot(index = 'Iteración', columns = 'Medida', values = 'Valor').reset_index(drop = True)

promedios = resultado_final.mean().to_frame(name = f'Promedios {modelo}')

print(promedios)


