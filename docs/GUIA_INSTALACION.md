# Guía de Instalación — EV3 Wine Quality

## Requisitos previos
- Python 3.13
- Git
- Cuenta y proyecto en Supabase
- pip

## 1. Clonar repositorio
```
git clone https://github.com/dongyah/EV3_SCY1101_wine_quality.git
cd EV3_SCY1101_wine_quality
```

## 2. Crear entorno virtual
```
python -m venv venv
venv\Scripts\activate
```

## 3. Instalar dependencias
```
pip install -r requirements.txt
```

## 4. Configurar variables de entorno
Crear archivo `.env` en la raíz del proyecto con:
```
SUPABASE_URL=tu_url_de_supabase
SUPABASE_KEY=tu_key_de_supabase
```

## 5. Ejecutar el pipeline ETL
```
python etl/main_etl.py
```
Esto integra las tres fuentes, valida esquemas y carga los datos a Supabase.

## 6. Levantar la API
```
python -m uvicorn api.main:app --reload
```
Disponible en http://127.0.0.1:8000

## 7. Levantar el Dashboard
En otra terminal, con la API corriendo:
```
python dashboards/app.py
```
Disponible en http://127.0.0.1:8050

## 8. Ejecutar pruebas automatizadas
```
python -m pytest -v
```
