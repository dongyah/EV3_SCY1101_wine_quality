import pytest
import joblib
import os
from fastapi.testclient import TestClient
import sys

# Agregar el directorio raíz al path para importar la API
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from api.main import app

# Cliente de prueba (no necesita que el servidor esté corriendo)
client = TestClient(app)

# Datos de un vino real típico para usar en los tests
VINO_REAL = {
    "fixed_acidity": 7.4,
    "volatile_acidity": 0.70,
    "citric_acid": 0.00,
    "residual_sugar": 1.9,
    "chlorides": 0.076,
    "free_sulfur_dioxide": 11.0,
    "total_sulfur_dioxide": 34.0,
    "density": 0.9978,
    "pH": 3.51,
    "sulphates": 0.56,
    "alcohol": 9.4
}


# TEST 1: Verificar que el modelo existe y carga correctamente
def test_modelo_carga():
    ruta = os.path.join(os.path.dirname(__file__), "..", "api", "wine_model.pkl")
    assert os.path.exists(ruta), "El archivo wine_model.pkl no existe"
    modelo = joblib.load(ruta)
    assert modelo is not None, "El modelo no se pudo cargar"


# TEST 2: Verificar que la API predice con datos válidos
def test_prediccion_valida():
    response = client.post("/predict", json=VINO_REAL)
    assert response.status_code == 200
    data = response.json()
    assert "calidad_predicha" in data
    assert data["calidad_predicha"] in ["bajo", "medio", "premium"]
    assert "confianza_porcentaje" in data
    assert 0 <= data["confianza_porcentaje"] <= 100


# TEST 3: Verificar que la API rechaza datos incompletos
def test_prediccion_datos_incompletos():
    datos_incompletos = {
        "fixed_acidity": 7.4,
        "volatile_acidity": 0.70
        # Faltan 9 parámetros
    }
    response = client.post("/predict", json=datos_incompletos)
    assert response.status_code == 422  # FastAPI devuelve 422 cuando faltan campos
