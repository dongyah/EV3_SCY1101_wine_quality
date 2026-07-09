from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import os

# 1. Inicializar la app
app = FastAPI(title="Wine Quality API", version="1.0.0")

# 2. Cargar los modelos entrenados
MODEL_DIR = os.path.dirname(__file__)
CLASSIFIER_PATH = os.path.join(MODEL_DIR, "wine_model.pkl")
REGRESSOR_PATH = os.path.join(MODEL_DIR, "wine_regressor.pkl")

modelo = joblib.load(CLASSIFIER_PATH)
regresor = joblib.load(REGRESSOR_PATH)

# 3. Definir el esquema de entrada (los 11 parámetros del vino)
class WineInput(BaseModel):
    fixed_acidity: float
    volatile_acidity: float
    citric_acid: float
    residual_sugar: float
    chlorides: float
    free_sulfur_dioxide: float
    total_sulfur_dioxide: float
    density: float
    pH: float
    sulphates: float
    alcohol: float

# Mapeo de campos del esquema Pydantic a los nombres de columnas de entrenamiento
COLUMNAS_MODELO = [
    "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
    "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
    "pH", "sulphates", "alcohol"
]

def wine_input_to_dataframe(wine: WineInput) -> pd.DataFrame:
    """Convierte el input de la API a un DataFrame de una fila estructurado para el modelo."""
    datos = {
        "fixed acidity": [wine.fixed_acidity],
        "volatile acidity": [wine.volatile_acidity],
        "citric acid": [wine.citric_acid],
        "residual sugar": [wine.residual_sugar],
        "chlorides": [wine.chlorides],
        "free sulfur dioxide": [wine.free_sulfur_dioxide],
        "total sulfur dioxide": [wine.total_sulfur_dioxide],
        "density": [wine.density],
        "pH": [wine.pH],
        "sulphates": [wine.sulphates],
        "alcohol": [wine.alcohol]
    }
    return pd.DataFrame(datos, columns=COLUMNAS_MODELO)

# 4. Endpoint de bienvenida
@app.get("/")
def root():
    return {"mensaje": "API de predicción de calidad de vino. Usa POST /predict y POST /predict_quality"}

# 5. Endpoint de predicción (Clasificación - Categoría)
@app.post("/predict")
def predict(wine: WineInput):
    try:
        # Convertir input a DataFrame con las columnas que el scaler y clasificador esperan
        features_df = wine_input_to_dataframe(wine)

        # Predecir
        prediccion = modelo.predict(features_df)[0]
        probabilidades = modelo.predict_proba(features_df)[0]
        confianza = round(float(max(probabilidades)) * 100, 2)

        return {
            "calidad_predicha": prediccion,
            "confianza_porcentaje": confianza
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# 6. Endpoint de predicción (Regresión - Calidad continua)
@app.post("/predict_quality")
def predict_quality(wine: WineInput):
    try:
        # Convertir input a DataFrame con las columnas que el regresor espera
        features_df = wine_input_to_dataframe(wine)

        # Predecir calidad continua
        prediccion_continua = regresor.predict(features_df)[0]
        return {
            "calidad_predicha_num": round(float(prediccion_continua), 2)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


