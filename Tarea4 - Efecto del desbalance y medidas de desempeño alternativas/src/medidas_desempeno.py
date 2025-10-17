from sklearn.metrics import (accuracy_score, balanced_accuracy_score,f1_score)
import pandas as pd
import numpy as np

def calcula_medidas(y_test, y_pred):
    '''
    Calculas las medidas de desempeño solicitadas en las instrucciones
    regresa una tabla con las medidas
    '''
    metrics = {

        'Accuracy': np.round(accuracy_score(y_test, y_pred), 4),
        'Balanced Accuracy': np.round(balanced_accuracy_score(y_test, y_pred),4),
        'F1 Score': np.round(f1_score(y_test, y_pred),4),
        'Accuraccy-BA': np.round(accuracy_score(y_test, y_pred) - balanced_accuracy_score(y_test, y_pred), 4)
    }

    return metrics






















'''
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,f1_score)
import pandas as pd
from sklearn.metrics import confusion_matrix

def calcula_medidas(y_test, y_pred):
    
    Calculas las medidas de desempeño solicitadas en las instrucciones
    regresa una tabla con las medidas
    
    metrics = {

        'Accuracy': accuracy_score(y_test, y_pred),
        'Balanced Accuracy': balanced_accuracy_score(y_test, y_pred),
        'F1 Score': f1_score(y_test, y_pred),
    }

    return pd.DataFrame(list(metrics.items()), columns = ['Medida', 'Valor'])
'''