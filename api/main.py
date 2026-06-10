from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import os

# 1. Inicializar la app
app = FastAPI(title="Wine Quality API", version="1.0.0")

# 2. Cargar el modelo entrenado
MODEL_PATH = os.path.join(os.path.dirname(__file__), "wine_model.pkl")
modelo = joblib.load(MODEL_PATH)

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

# 4. Endpoint de bienvenida
@app.get("/")
def root():
    return {"mensaje": "API de predicción de calidad de vino. Usa POST /predict"}

# 5. Endpoint de predicción
@app.post("/predict")
def predict(wine: WineInput):
    try:
        # Convertir input a array para el modelo
        features = np.array([[
            wine.fixed_acidity,
            wine.volatile_acidity,
            wine.citric_acid,
            wine.residual_sugar,
            wine.chlorides,
            wine.free_sulfur_dioxide,
            wine.total_sulfur_dioxide,
            wine.density,
            wine.pH,
            wine.sulphates,
            wine.alcohol
        ]])

        # Predecir
        prediccion = modelo.predict(features)[0]
        probabilidades = modelo.predict_proba(features)[0]
        confianza = round(float(max(probabilidades)) * 100, 2)

        return {
            "calidad_predicha": prediccion,
            "confianza_porcentaje": confianza
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
