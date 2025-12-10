import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

def imprime_ROC_AUC(y_true, y_pred_proba):

    '''
    Imprime la curva ROC-AUC
    '''

    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    auc = roc_auc_score(y_true, y_pred_proba)

    
    plt.figure(figsize=(9,9))
    plt.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {auc:.2f})')
    plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Curva ROC')
    plt.legend()
    plt.grid(True)
    plt.show()
