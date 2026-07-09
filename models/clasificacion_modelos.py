import os
import sys
import logging
import joblib
import shutil
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE

# Configuración de logging profesional
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("models/clasificacion_modelos.log", encoding="utf-8")
    ]
)
logger = logging.getLogger(__name__)

# Asegurar que existe el directorio de logs / models
os.makedirs("models", exist_ok=True)

def clasificar_calidad(q: int) -> str:
    """Clasifica la calidad numérica en categorías: bajo, medio o premium."""
    if q <= 4:
        return "bajo"
    elif q <= 6:
        return "medio"
    else:
        return "premium"

def entrenar_clasificacion():
    """Ejecuta el pipeline de modelamiento para clasificación de vinos."""
    logger.info("Iniciando pipeline de entrenamiento para clasificación...")
    
    # 1. Cargar dataset
    ruta_csv = "data/winequality_clean.csv"
    if not os.path.exists(ruta_csv):
        logger.error(f"No se encontró el archivo de datos en {ruta_csv}")
        sys.exit(1)
        
    try:
        df = pd.read_csv(ruta_csv)
        logger.info(f"Dataset cargado con éxito. Filas: {len(df)}, Columnas: {len(df.columns)}")
    except Exception as e:
        logger.error(f"Error al leer el dataset: {e}")
        sys.exit(1)

    logger.info(f"Columnas detectadas en el CSV: {list(df.columns)}")
    
    # Crear variable categórica objetivo
    df["categoria"] = df["quality"].apply(clasificar_calidad)
    
    # Separar predictores y variable objetivo
    X = df.drop(columns=["quality", "categoria"])
    y = df["categoria"]
    
    # Split entrenamiento y prueba (80/20) estratificado por el desbalance
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    logger.info(f"Split de datos completado. Train: {len(X_train)} filas, Test: {len(X_test)} filas")
    logger.info(f"Distribución original de clases en entrenamiento:\n{y_train.value_counts()}")

    # Aplicar SMOTE en entrenamiento para balancear clases
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    logger.info(f"Distribución tras SMOTE en entrenamiento:\n{pd.Series(y_train_res).value_counts()}")

    # 2. Configurar modelos y pipelines
    pipe_rf = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(random_state=42))
    ])
    
    pipe_lr = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=42))
    ])

    # Parámetros para GridSearchCV
    param_grid_rf = {
        "clf__n_estimators": [100, 200],
        "clf__max_depth": [10, 15, None],
        "clf__min_samples_split": [2, 5]
    }

    param_grid_lr = {
        "clf__C": [0.01, 0.1, 1.0, 10.0],
        "clf__solver": ["lbfgs", "saga"]
    }

    # 3. Optimización con GridSearchCV
    logger.info("Ejecutando GridSearchCV para RandomForestClassifier...")
    grid_rf = GridSearchCV(pipe_rf, param_grid_rf, cv=5, scoring="f1_macro", n_jobs=-1)
    grid_rf.fit(X_train_res, y_train_res)
    best_rf = grid_rf.best_estimator_
    logger.info(f"Mejor configuración RF: {grid_rf.best_params_} (F1-macro: {grid_rf.best_score_:.4f})")

    logger.info("Ejecutando GridSearchCV para LogisticRegression...")
    grid_lr = GridSearchCV(pipe_lr, param_grid_lr, cv=5, scoring="f1_macro", n_jobs=-1)
    grid_lr.fit(X_train_res, y_train_res)
    best_lr = grid_lr.best_estimator_
    logger.info(f"Mejor configuración LR: {grid_lr.best_params_} (F1-macro: {grid_lr.best_score_:.4f})")

    # 4. Evaluación en el conjunto de prueba independiente
    logger.info("Evaluando RandomForestClassifier en test...")
    y_pred_rf = best_rf.predict(X_test)
    acc_rf = accuracy_score(y_test, y_pred_rf)
    report_rf = classification_report(y_test, y_pred_rf, zero_division=0)
    cm_rf = confusion_matrix(y_test, y_pred_rf, labels=["bajo", "medio", "premium"])
    logger.info(f"RF Accuracy: {acc_rf:.4f}\nReporte:\n{report_rf}\nMatriz de confusión:\n{cm_rf}")

    logger.info("Evaluando LogisticRegression en test...")
    y_pred_lr = best_lr.predict(X_test)
    acc_lr = accuracy_score(y_test, y_pred_lr)
    report_lr = classification_report(y_test, y_pred_lr, zero_division=0)
    cm_lr = confusion_matrix(y_test, y_pred_lr, labels=["bajo", "medio", "premium"])
    logger.info(f"LR Accuracy: {acc_lr:.4f}\nReporte:\n{report_lr}\nMatriz de confusión:\n{cm_lr}")

    # Seleccionar el mejor modelo
    mejor_modelo = best_rf
    nombre_mejor = "RandomForestClassifier"
    if acc_lr > acc_rf:
        mejor_modelo = best_lr
        nombre_mejor = "LogisticRegression"
        
    logger.info(f"Modelo seleccionado como el mejor: {nombre_mejor} con accuracy de {max(acc_rf, acc_lr):.4f}")

    # 5. Serializar / Guardar el modelo
    os.makedirs("api", exist_ok=True)
    ruta_guardado = "api/wine_classifier.pkl"
    ruta_retrocompatibilidad = "api/wine_model.pkl"
    
    try:
        joblib.dump(mejor_modelo, ruta_guardado)
        logger.info(f"Modelo guardado en {ruta_guardado}")
        shutil.copy(ruta_guardado, ruta_retrocompatibilidad)
        logger.info(f"Copia de retrocompatibilidad guardada en {ruta_retrocompatibilidad}")
    except Exception as e:
        logger.error(f"Error al serializar el modelo: {e}")
        sys.exit(1)

if __name__ == "__main__":
    entrenar_clasificacion()
