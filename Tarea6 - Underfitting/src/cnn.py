import numpy as np
import pandas as pd

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