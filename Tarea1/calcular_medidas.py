from sklearn.metrics import (accuracy_score, recall_score, balanced_accuracy_score, precision_score, f1_score, matthews_corrcoef, roc_auc_score)

def calcula_medidas(y_test, y_pred):
    '''
    Calculas las medidas de desempeño solicitadas en las instrucciones
    regresa una tabla con las medidas
    '''

    accuracy = accuracy_score(y_test, y_pred)
    error_rate = 1 - accuracy
    recall = recall_score(y_test, y_pred)
    specificity = recall_score(y_test, y_pred, pos_label = 0)
    bacc = balanced_accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)

    metrics_table = [
        ['Accuracy', accuracy],
        ['Error Rate', error_rate],
        ['Recall (Sensibilidad)', recall],
        ['Specificity', specificity],
        ['Balanced Accuracy', bacc],
        ['Precision', precision],
        ['F1 Score', f1],
        ['MCC', mcc],
    ]

    return metrics_table