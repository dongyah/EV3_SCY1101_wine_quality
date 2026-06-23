import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
import joblib
import os
import logging
import sys


## Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("etl/smote_pipeline.log", encoding="utf-8")
    ]
)
logger = logging.getLogger(__name__)

#validacion de esquema
COLUMNAS_ESPERADAS = {
    "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
    "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
    "pH", "sulphates", "alcohol", "quality"
}

RUTA_CSV = "data/winequality_clean.csv"

try:
    df = pd.read_csv(RUTA_CSV)
    logger.info(f"CSV cargado: {len(df)} filas, {len(df.columns)} columnas")
except FileNotFoundError:
    logger.error(f"No se encontró el archivo: {RUTA_CSV}")
    sys.exit(1)
except Exception as e:
    logger.error(f"Error al leer el CSV: {e}")
    sys.exit(1)

columnas_presentes = set(df.columns)
columnas_faltantes = COLUMNAS_ESPERADAS - columnas_presentes
if columnas_faltantes:
    logger.error(f"Columnas faltantes: {columnas_faltantes}")
    sys.exit(1)
logger.info("Validación de esquema: OK")

nulos = df.isnull().sum().sum()
if nulos > 0:
    logger.warning(f"{nulos} valores nulos encontrados — se eliminarán")
    df = df.dropna()
    logger.info(f"Filas tras eliminar nulos: {len(df)}")
else:
    logger.info("Sin valores nulos")


def clasificar_calidad(q):
    if q <= 4:
        return "bajo"
    elif q <= 6:
        return "medio"
    else:
        return "premium"

try:
    df["categoria"] = df["quality"].apply(clasificar_calidad)
    logger.info(f"Distribución de categorías:\n{df['categoria'].value_counts().to_string()}")
except Exception as e:
    logger.error(f"Error al clasificar calidad: {e}")
    sys.exit(1)

try:
    X = df.drop(columns=["quality", "categoria"])
    y = df["categoria"]
    logger.info(f"Features: {list(X.columns)}")
except Exception as e:
    logger.error(f"Error al separar features: {e}")
    sys.exit(1)

try:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    logger.info(f"Split: {len(X_train)} entrenamiento / {len(X_test)} test")
except Exception as e:
    logger.error(f"Error en train_test_split: {e}")
    sys.exit(1)

try:
    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
    logger.info(f"Distribución tras SMOTE:\n{pd.Series(y_train_bal).value_counts().to_string()}")
except Exception as e:
    logger.error(f"Error en SMOTE: {e}")
    sys.exit(1)

try:
    modelo = RandomForestClassifier(n_estimators=200, random_state=42)
    modelo.fit(X_train_bal, y_train_bal)
    logger.info("Modelo RandomForest entrenado correctamente")
except Exception as e:
    logger.error(f"Error al entrenar el modelo: {e}")
    sys.exit(1)

try:
    y_pred = modelo.predict(X_test)
    reporte = classification_report(y_test, y_pred)
    logger.info(f"Resultados en test set real:\n{reporte}")
except Exception as e:
    logger.error(f"Error en evaluación: {e}")
    sys.exit(1)

try:
    os.makedirs("api", exist_ok=True)
    joblib.dump(modelo, "api/wine_model.pkl")
    logger.info("Modelo guardado en api/wine_model.pkl")
except Exception as e:
    logger.error(f"Error al guardar el modelo: {e}")
    sys.exit(1)

try:
    os.makedirs("data", exist_ok=True)
    df_balanced = X_train_bal.copy()
    df_balanced["categoria"] = y_train_bal.values
    df_balanced.to_csv("data/winequality_balanced.csv", index=False)
    logger.info("Dataset balanceado guardado en data/winequality_balanced.csv")
except Exception as e:
    logger.error(f"Error al guardar dataset balanceado: {e}")
    sys.exit(1)

logger.info("Pipeline SMOTE completado exitosamente")