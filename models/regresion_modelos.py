import os
import sys
import logging
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Configuración de logging profesional
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("models/regresion_modelos.log", encoding="utf-8")
    ]
)
logger = logging.getLogger(__name__)

# Asegurar que existe el directorio de logs / models
os.makedirs("models", exist_ok=True)

def entrenar_regresion():
    """Ejecuta el pipeline de modelamiento para regresión de la calidad del vino."""
    logger.info("Iniciando pipeline de entrenamiento para regresión...")
    
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
    
    # Separar predictores y variable objetivo (calidad numérica continua)
    X = df.drop(columns=["quality"])
    y = df["quality"]
    
    # Split entrenamiento y prueba (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    logger.info(f"Split de datos completado. Train: {len(X_train)} filas, Test: {len(X_test)} filas")

    # 2. Configurar modelos y pipelines
    # StandardScaler es importante para Ridge (regularización L2 sensible a escalas)
    pipe_rf = Pipeline([
        ("scaler", StandardScaler()),
        ("reg", RandomForestRegressor(random_state=42))
    ])
    
    pipe_ridge = Pipeline([
        ("scaler", StandardScaler()),
        ("reg", Ridge(random_state=42))
    ])

    # Parámetros para GridSearchCV
    param_grid_rf = {
        "reg__n_estimators": [100, 200],
        "reg__max_depth": [10, 15, None],
        "reg__min_samples_split": [2, 5]
    }

    param_grid_ridge = {
        "reg__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]
    }

    # 3. Optimización con GridSearchCV
    logger.info("Ejecutando GridSearchCV para RandomForestRegressor...")
    grid_rf = GridSearchCV(pipe_rf, param_grid_rf, cv=5, scoring="neg_mean_squared_error", n_jobs=-1)
    grid_rf.fit(X_train, y_train)
    best_rf = grid_rf.best_estimator_
    logger.info(f"Mejor configuración RF: {grid_rf.best_params_} (MSE validación: {-grid_rf.best_score_:.4f})")

    logger.info("Ejecutando GridSearchCV para Ridge...")
    grid_ridge = GridSearchCV(pipe_ridge, param_grid_ridge, cv=5, scoring="neg_mean_squared_error", n_jobs=-1)
    grid_ridge.fit(X_train, y_train)
    best_ridge = grid_ridge.best_estimator_
    logger.info(f"Mejor configuración Ridge: {grid_ridge.best_params_} (MSE validación: {-grid_ridge.best_score_:.4f})")

    # 4. Evaluación en el conjunto de prueba independiente
    logger.info("Evaluando RandomForestRegressor en test...")
    y_pred_rf = best_rf.predict(X_test)
    mae_rf = mean_absolute_error(y_test, y_pred_rf)
    rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
    r2_rf = r2_score(y_test, y_pred_rf)
    logger.info(f"RF Regressor -> MAE: {mae_rf:.4f}, RMSE: {rmse_rf:.4f}, R2: {r2_rf:.4f}")

    logger.info("Evaluando Ridge en test...")
    y_pred_ridge = best_ridge.predict(X_test)
    mae_ridge = mean_absolute_error(y_test, y_pred_ridge)
    rmse_ridge = np.sqrt(mean_squared_error(y_test, y_pred_ridge))
    r2_ridge = r2_score(y_test, y_pred_ridge)
    logger.info(f"Ridge Regressor -> MAE: {mae_ridge:.4f}, RMSE: {rmse_ridge:.4f}, R2: {r2_ridge:.4f}")

    # Seleccionar el mejor modelo (menor RMSE en test)
    mejor_modelo = best_rf
    nombre_mejor = "RandomForestRegressor"
    if rmse_ridge < rmse_rf:
        mejor_modelo = best_ridge
        nombre_mejor = "Ridge"
        
    logger.info(f"Modelo seleccionado como el mejor: {nombre_mejor} con RMSE de {min(rmse_rf, rmse_ridge):.4f}")

    # 5. Serializar / Guardar el modelo de regresión
    os.makedirs("api", exist_ok=True)
    ruta_guardado = "api/wine_regressor.pkl"
    
    try:
        joblib.dump(mejor_modelo, ruta_guardado)
        logger.info(f"Modelo de regresión guardado en {ruta_guardado}")
    except Exception as e:
        logger.error(f"Error al serializar el modelo: {e}")
        sys.exit(1)

if __name__ == "__main__":
    entrenar_regresion()
