# EV3 SCY1101 — Sistema de Predicción de Calidad de Vino Tinto

Proyecto individual de la estudiante Belén Toloza para la asignatura SCY1101.  

Este proyecto implementa una arquitectura end-to-end para el procesamiento, almacenamiento, predicción y visualización de la calidad de vinos tintos. El sistema integra tres fuentes de datos (CSV, API REST propia y base de datos en la nube) y clasifica cada muestra en tres categorías de negocio: **bajo** (≤4), **medio** (5–6) y **premium** (≥7).

## Arquitectura del Sistema

El sistema está compuesto por cuatro capas independientes:

1. **Pipeline ETL** (`etl/`): Orquesta la extracción de las 3 fuentes, balancea clases con SMOTE y entrena un modelo `RandomForestClassifier` (200 estimadores), guardado en `api/wine_model.pkl`.
2. **Almacenamiento** (`data` / Supabase): Base de datos en la nube (PostgreSQL vía Supabase), tabla `vinos`, que persiste los registros limpios e integrados.
3. **Capa de Negocio / API REST** (`api/`): Desarrollada con **FastAPI** y ejecutada mediante Uvicorn. Expone el endpoint `/predict`, que recibe las propiedades fisicoquímicas del vino y retorna la calidad predicha con su nivel de confianza.
4. **Capa de Presentación / Dashboard** (`dashboards/`): Interfaz interactiva construida con **Plotly Dash**, con tres vistas diferenciadas por audiencia (ejecutiva, técnica, operativa), que consume la API mediante peticiones HTTP.

## Estructura de carpetas

- `/etl/` — Scripts del pipeline ETL (SMOTE, carga a Supabase, orquestador)
- `/api/` — API RESTful (FastAPI) y modelo entrenado
- `/dashboards/` — Dashboard Plotly Dash
- `/data/` — Datos originales, balanceados e integrados
- `/docs/` — Documentación técnica, manuales, guía de instalación
- `/tests/` — Tests automatizados (pytest)
- `/docker/` — Dockerfiles optimizados y orquestación con docker-compose.

## Requisitos Previos

- Python 3.13
- Cuenta y proyecto en [Supabase](https://supabase.com/)
- Git

## Instalación

```bash
pip install -r requirements.txt
```

## Variables de entorno

Crear archivo `.env` en la raíz con:

```
SUPABASE_URL=tu_url_de_supabase
SUPABASE_KEY=tu_key_de_supabase
```

## Instrucciones de Ejecución

```bash
# 1. Entrenar modelo y generar dataset balanceado
python etl/smote_wine.py

# 2. Cargar datos a Supabase
python etl/cargar_supabase.py

# 3. Levantar la API (puerto 8000)
python -m uvicorn api.main:app --reload

# 4. Ejecutar orquestador ETL (integra las 3 fuentes)
python etl/main_etl.py

# 5. Levantar el Dashboard (puerto 8050)
python dashboards/app.py

# 6. Correr tests
python -m pytest
```

Accesos:
- Dashboard: http://localhost:8050
- Documentación interactiva de la API (Swagger): http://localhost:8000/docs

## 🐳 Despliegue con Docker (Fase 3)

Todo el entorno ha sido containerizado. Para levantar la API, el Dashboard y ejecutar el ETL automáticamente en contenedores:

```bash
cd docker
docker-compose up --build -d
```

Servicios levantados:
- **Dashboard:** http://localhost:8050
- **API REST (Docs):** http://localhost:8000/docs

## Documentación adicional

- [Documentación de API](docs/API.md)
- [Manual de usuario](docs/MANUAL_USUARIO.md)
- [Guía de instalación y despliegue](docs/GUIA_INSTALACION.md)

## 🎥 Video de Exposición (Fase 3)

🔗 **Enlace al video de Google Drive:** [INSERTE_LINK_AQUI] *(¡Importante: Recuerda habilitar los permisos para que cualquier persona con el enlace pueda verlo!)*

Comandos para aplicar y subir (PowerShell, en la raíz del repo):

```powershell
git pull origin main
notepad README.md
```