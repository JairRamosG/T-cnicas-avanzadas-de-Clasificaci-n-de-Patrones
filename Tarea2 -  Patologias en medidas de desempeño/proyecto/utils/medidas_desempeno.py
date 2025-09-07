from sklearn.metrics import (accuracy_score, recall_score, balanced_accuracy_score, precision_score, f1_score, matthews_corrcoef)
import pandas as pd
from sklearn.metrics import confusion_matrix

def calcula_medidas(y_test, y_pred, clase_positiva):
    '''
    Calcula las medidas de desempeño usando la prediccion generada y las etiquetas reales
    tomando en cuenta cuál es la clase positiva
    '''

    cm = confusion_matrix(y_test, y_pred, labels=[1-clase_positiva, clase_positiva])
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn+fp) > 0 else 0

    
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
