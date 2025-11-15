import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

def CNN(X, y):

    X = np.array(X)
    y = np.array(y)
    
    Sx, Sy = [], []
    clases = np.unique(y)
    for c in clases:
        idx = np.where(y == c)[0][0]
        Sx.append(X[idx])
        Sy.append(y[idx])

    Sx_array = np.array(Sx)
    Sy_array = np.array(Sy)
    
    changed = True    
    while changed:
        changed = False
        
        indices = np.random.permutation(len(X))
        
        for idx in indices:
            xi, yi = X[idx], y[idx]
            
            if any(np.array_equal(xi, s_point) for s_point in Sx_array):
                continue

            dists = np.linalg.norm(Sx_array - xi, axis=1)
            idx_nn = np.argmin(dists)
            
            if Sy_array[idx_nn] != yi:
                Sx_array = np.vstack([Sx_array, xi])
                Sy_array = np.append(Sy_array, yi)
                changed = True
    
    df_reducido = pd.DataFrame(Sx_array)
    df_reducido['target'] = Sy_array
    
    return df_reducido


def condensed_nearest_neighbors(X, y):
    """
    Implementación del algoritmo Condensed Nearest Neighbors (CNN).
    X: matriz de características (numpy array)
    y: vector de etiquetas (numpy array)
    Retorna: subconjunto condensado (X_cond, y_cond)
    """

    X = np.array(X)
    y = np.array(y)

    rng = np.random.default_rng()
    idx = rng.integers(0, len(X))
    
    X_cond = np.array([X[idx]])
    y_cond = np.array([y[idx]])

    changed = True

    while changed:
        changed = False

        for xi, yi in zip(X, y):
            knn = KNeighborsClassifier(n_neighbors=1)
            knn.fit(X_cond, y_cond)

            y_pred = knn.predict([xi])[0]

            if y_pred != yi:
                X_cond = np.vstack([X_cond, xi])
                y_cond = np.append(y_cond, yi)
                changed = True

    df_cond = pd.DataFrame(X_cond)
    df_cond['target'] = y_cond
    
    return df_cond
