# API de Predicción de Calidad de Vino

## Descripción
API REST construida con FastAPI que expone un modelo RandomForest (`wine_model.pkl`) para predecir la categoría de calidad de un vino tinto a partir de sus propiedades fisicoquímicas.

## Base URL
```
http://localhost:8000
```

## Ejecución local
```
python -m uvicorn api.main:app --reload --port 8000
```

## Endpoints

### `GET /`
Endpoint de bienvenida / health check.

**Respuesta 200:**
```json
{
  "mensaje": "API de predicción de calidad de vino. Usa POST /predict"
}
```

---

### `POST /predict`
Predice la categoría de calidad de un vino.

**Request body (`application/json`):**

| Campo | Tipo | Descripción |
|---|---|---|
| fixed_acidity | float | Acidez fija |
| volatile_acidity | float | Acidez volátil |
| citric_acid | float | Ácido cítrico |
| residual_sugar | float | Azúcar residual |
| chlorides | float | Cloruros |
| free_sulfur_dioxide | float | Dióxido de azufre libre |
| total_sulfur_dioxide | float | Dióxido de azufre total |
| density | float | Densidad |
| pH | float | pH (nota: capital H, distinto del `ph` usado en CSV/Supabase) |
| sulphates | float | Sulfatos |
| alcohol | float | Grado alcohólico |

**Ejemplo de request:**
```json
{
  "fixed_acidity": 7.4,
  "volatile_acidity": 0.7,
  "citric_acid": 0.0,
  "residual_sugar": 1.9,
  "chlorides": 0.076,
  "free_sulfur_dioxide": 11,
  "total_sulfur_dioxide": 34,
  "density": 0.9978,
  "pH": 3.51,
  "sulphates": 0.56,
  "alcohol": 9.4
}
```

**Respuesta 200:**
```json
{
  "calidad_predicha": "medio",
  "confianza_porcentaje": 87.5
}
```

| Campo | Tipo | Descripción |
|---|---|---|
| calidad_predicha | string | Categoría predicha: `bajo`, `medio` o `premium` |
| confianza_porcentaje | float | Probabilidad máxima entre las clases (0-100) |

**Respuesta 500 (error interno, ej. dato inválido o fallo del modelo):**
```json
{
  "detail": "<mensaje de excepción>"
}
```

**Ejemplo curl (PowerShell):**
```
curl -X POST http://localhost:8000/predict `
  -H "Content-Type: application/json" `
  -d '{\"fixed_acidity\":7.4,\"volatile_acidity\":0.7,\"citric_acid\":0.0,\"residual_sugar\":1.9,\"chlorides\":0.076,\"free_sulfur_dioxide\":11,\"total_sulfur_dioxide\":34,\"density\":0.9978,\"pH\":3.51,\"sulphates\":0.56,\"alcohol\":9.4}'
```
