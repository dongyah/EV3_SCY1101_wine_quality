# EV3 SCY1101 — Sistema de Predicción de Calidad de Vino Tinto

Proyecto individual de la estudiante Belén Toloza para la asignatura SCY1101.  

Este proyecto implementa una arquitectura de nivel industrial (*End-to-End*) de MLOps para el procesamiento, almacenamiento, predicción y visualización de la calidad de vinos tintos. El sistema integra tres fuentes de datos (CSV, API REST propia y Supabase Postgres) y permite tanto la clasificación categórica en tres clases de negocio (**bajo** ≤4, **medio** 5–6, **premium** ≥7) como la predicción continua de la calidad (escala 0-10) mediante regresión.

## Arquitectura del Sistema

El sistema está compuesto por las siguientes capas modulares:

1. **Pipeline ETL** (`etl/`): Orquesta la extracción y limpieza estricta (coerción a numérico, validación y descarte con `dropna`) de las 3 fuentes de datos. Implementa paginación en lotes de 1000 registros para la extracción en Supabase.
2. **Modelamiento Predictivo** (`models/`):
   - **Clasificación (`models/clasificacion_modelos.py`)**: Entrena y compara RandomForest (Accuracy: 77.57%) y LogisticRegression usando balanceo SMOTE y GridSearchCV ($K=5$). Guarda el modelo en `api/wine_classifier.pkl` (y `wine_model.pkl`).
   - **Regresión (`models/regresion_modelos.py`)**: Entrena y compara RandomForestRegressor (RMSE: 0.6160) y Ridge Regressor (RMSE: 0.6554) bajo GridSearchCV ($K=5$) para amortiguar la multicolinealidad con penalización L2. Guarda el regresor en `api/wine_regressor.pkl`.
3. **Capa de Negocio / API REST** (`api/`): Desarrollada con **FastAPI**. Expone:
   - `POST /predict`: Clasificación categórica y confianza.
   - `POST /predict_quality`: Estimación de calidad numérica continua (regresión).
4. **Capa de Presentación / Dashboard** (`dashboards/`): Interfaz interactiva construida con **Plotly Dash**, que consume la API mediante peticiones HTTP.
5. **Integración Continua** (`.github/workflows/ci.yml`): Workflow automatizado en GitHub Actions que entrena dinámicamente los modelos en el runner y ejecuta con éxito 5/5 pruebas unitarias en `pytest`.

## Estructura de carpetas

- `/etl/` — Scripts del pipeline ETL (limpieza, carga a Supabase y orquestador)
- `/models/` — Scripts de entrenamiento de clasificación y regresión con justificación teórica
- `/api/` — API RESTful (FastAPI) y modelos serializados (`.pkl`)
- `/dashboards/` — Dashboard Plotly Dash
- `/data/` — Datos originales, balanceados e integrados
- `/docs/` — Documentación técnica, manuales y guías
- `/tests/` — Tests unitarios automatizados (pytest)
- `/docker/` y `docker-compose.yml` — Archivos de orquestación con Docker

## Requisitos Previos

- Python 3.13
- Cuenta y proyecto en [Supabase](https://supabase.com/)
- Git / Docker

## Variables de entorno

Crear archivo `.env` en la raíz con:

```env
SUPABASE_URL=tu_url_de_supabase
SUPABASE_KEY=tu_key_de_supabase
```

## Instrucciones de Ejecución

```bash
# 1. Entrenar y guardar modelos predictivos (clasificación y regresión)
python -m models.clasificacion_modelos
python -m models.regresion_modelos

# 2. Cargar datos iniciales a Supabase
python etl/cargar_supabase.py

# 3. Levantar la API en local (puerto 8000)
python -m uvicorn api.main:app --reload

# 4. Ejecutar orquestador ETL (integra las 3 fuentes consultando la API en caliente)
python etl/main_etl.py

# 5. Levantar el Dashboard (puerto 8050)
python dashboards/app.py

# 6. Correr pruebas unitarias locales
python -m pytest tests/ -v
```

Accesos:
- Dashboard: http://localhost:8050
- Documentación interactiva de la API (Swagger): http://localhost:8000/docs

## 🐳 Despliegue con Docker

Todo el entorno está containerizado. Para levantar la API, el Dashboard y el ETL de forma automática:

```bash
cd docker
docker-compose up --build -d
```

## Documentación adicional

- [Documentación de API](docs/API.md)
- [Manual de usuario](docs/MANUAL_USUARIO.md)
- [Guía de instalación y despliegue](docs/GUIA_INSTALACION.md)
- [Ficha Técnica Consolidada MLOps](C:/Users/belen/.gemini/antigravity-ide/brain/72d5448b-8c9e-4e3a-9b05-5064ce31134b/informe_tecnico_ejecutivo.md)
