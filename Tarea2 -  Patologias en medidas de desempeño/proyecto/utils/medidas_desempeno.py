from sklearn.metrics import (accuracy_score, recall_score, balanced_accuracy_score, precision_score, f1_score, matthews_corrcoef)
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path


def guarda_matriz(cm, algoritmo, clase_positiva):
    '''
    Utiliza las prediciónes para guardar una imagen de las matrices de confusión
    '''

    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    nombre_archivo = f'{algoritmo}_positiva_{clase_positiva}'
    ruta= os.path.join(BASE_DIR, 'imagenes', nombre_archivo)

    cm_vis = [[cm[1,1], cm[1,0]],
              [cm[0,1], cm[0,0]]]

    plt.figure(figsize=(6,5))
    sns.heatmap(cm_vis, annot=True, fmt='d', cmap='Blues', xticklabels=['Pos', 'Neg'], yticklabels=['Pos', 'Neg'], cbar=False)

    plt.xlabel('Prediccion')
    plt.ylabel('Real')
    plt.title(nombre_archivo)
    plt.savefig(ruta, dpi=300, bbox_inches='tight')



def calcula_medidas(y_test, y_pred, clase_positiva, algoritmo):
    '''
    Calcula las medidas de desempeño usando la prediccion generada y las etiquetas reales
    tomando en cuenta cuál es la clase positiva
    '''

    cm = confusion_matrix(y_test, y_pred, labels=[0,1])
    tn, fp, fn, tp = cm.ravel()
    
    guarda_matriz(cm, algoritmo, clase_positiva)

    if clase_positiva == 1:
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    else:
        specificity = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    metrics = {

        'Accuracy': accuracy_score(y_test, y_pred),
        'Error Rate': 1 - accuracy_score(y_test, y_pred),
        'Recall (Sensitivity)': recall_score(y_test, y_pred, pos_label = clase_positiva),
        'Specificity': specificity,
        'Balanced Accuracy': balanced_accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, pos_label = clase_positiva),
        'F1 Score': f1_score(y_test, y_pred, pos_label = clase_positiva),
        'MCC': matthews_corrcoef(y_test, y_pred)
    }

    return pd.DataFrame(list(metrics.items()), columns = ['Medida', 'Valor'])
