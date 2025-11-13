import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap

def visualiza_svm(modelo, X, y, titulo, palette='mako'):
    # Crear malla
    x_min, x_max = X.iloc[:, 0].min() - 0.5, X.iloc[:, 0].max() + 0.5
    y_min, y_max = X.iloc[:, 1].min() - 0.5, X.iloc[:, 1].max() + 0.5
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300))

    # Predicción sobre la malla
    Z = modelo.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Obtener colores desde Seaborn y crear un colormap
    clases_unicas = np.unique(y)
    colores = sns.color_palette(palette, n_colors=len(clases_unicas))
    cmap = ListedColormap(colores)

    plt.figure(figsize=(8,6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=cmap)

    # Asegurar formato del target
    y = pd.Series(y).reset_index(drop=True)
    X = X.reset_index(drop=True)

    # --- Mapeo automático si las clases están codificadas ---
    # (ajusta si tus etiquetas son distintas)
    mapeo = {0: 'versicolor', 1: 'virginica', 2: 'setosa'}
    y_nombres = y.map(mapeo) if y.dtype != object else y

    clases = np.unique(y_nombres)
    colores_puntos = sns.color_palette(palette, n_colors=len(clases))

    # Graficar puntos de cada clase con etiquetas originales
    for i, clase in enumerate(clases):
        plt.scatter(
            X[y_nombres == clase].iloc[:, 0],
            X[y_nombres == clase].iloc[:, 1],
            color=colores_puntos[i],
            edgecolor='k',
            s=60,
            label=clase
        )

    plt.xlabel(X.columns[0])
    plt.ylabel(X.columns[1])
    plt.title(titulo)
    plt.legend(title='Clases', loc='upper right', fontsize=10)
    plt.show()