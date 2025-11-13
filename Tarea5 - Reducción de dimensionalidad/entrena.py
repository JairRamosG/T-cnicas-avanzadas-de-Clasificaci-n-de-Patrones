import pandas as pd
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

from src.medidas_desempeno_biclase import calcula_medidas_biclase
from src.ROC_AUC import imprime_ROC_AUC
import time


def entrenamiento(df, clase):

    X = df.drop(columns = ['CLASS'])
    y = df['CLASS']

    # SVM
    svm_model = svm.SVC()
    cv_vals = [5, 10]
    resultados_svm = []

    params = {'kernel' : ['rbf'],
        'C': [1, 3],
        'gamma': [0.0001, 0.0002]}

    medidas = {
            'accuracy': 'accuracy',
            'precision': 'precision',
            'recall': 'recall',
            'f1': 'f1',
            'ROC_AUC': 'roc_auc'}

    for cval in cv_vals:    
        grid = GridSearchCV(
            estimator = svm_model,
            param_grid = params,
            cv = cval,
            scoring = medidas,
            refit = 'ROC_AUC',
            n_jobs = -1)

        grid.fit(X, y)
        resultados_svm.append({
            'cv': cval,
            'best_score': grid.best_score_,
            'best_params': grid.best_params_, 
            'medidas': grid.cv_results_})
        
    df_resultados_svm = pd.DataFrame(resultados_svm)
    
    # KNN
    knn_model = KNeighborsClassifier()
    cv_vals = [5, 10]
    resultados_knn = []

    params = {'n_neighbors':[11, 13, 15]}

    medidas = {
            'accuracy': 'accuracy',
            'precision': 'precision',
            'recall': 'recall',
            'f1': 'f1',
            'ROC_AUC': 'roc_auc'}

    for cval in cv_vals:    
        grid = GridSearchCV(
            estimator = knn_model,
            param_grid = params,
            cv = cval,
            scoring = medidas,
            refit = 'ROC_AUC',
            n_jobs = -1)

        grid.fit(X, y)
        resultados_knn.append({
            'cv': cval,
            'best_score': grid.best_score_,
            'best_params': grid.best_params_, 
            'medidas': grid.cv_results_})
        
    df_resultados_knn = pd.DataFrame(resultados_knn)

            