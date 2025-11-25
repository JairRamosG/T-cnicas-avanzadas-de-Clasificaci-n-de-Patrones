from sklearn.preprocessing import StandardScaler, MinMaxScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import pandas as pd

def preparar_dataset(
    df,
    tipo_escalado="standard",
    imputar=True):

    columnas_numericas = df.select_dtypes(include=["int64", "float64"]).columns
    columnas_categoricas = df.select_dtypes(include=["object", "category"]).columns

    if tipo_escalado == "standard":
        escalador = StandardScaler()
    elif tipo_escalado == "minmax":
        escalador = MinMaxScaler()
    else:
        raise ValueError("tipo_escalado debe ser 'standard' o 'minmax'")

    pipeline_numerico = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean")) if imputar else ("passthrough", "passthrough"),
        ("scaler", escalador)
    ])

    pipeline_categorico = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")) if imputar else ("passthrough", "passthrough"),
        ("encoder", OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1
        ))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", pipeline_numerico, columnas_numericas),
            ("cat", pipeline_categorico, columnas_categoricas)
        ]
    )

    X_procesado = preprocessor.fit_transform(df)

    nombres_finales = list(columnas_numericas) + list(columnas_categoricas)

    df_final = pd.DataFrame(X_procesado, columns=nombres_finales)

    return df_final
