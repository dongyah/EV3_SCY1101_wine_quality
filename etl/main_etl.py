"""
main_etl.py — Orquestador ETL End-to-End
Fuente 1: CSV local (winequality_clean.csv)
Fuente 2: API RESTful FastAPI (/predict)
Fuente 3: Supabase PostgreSQL (tabla 'vinos')
"""

import os
import logging
import pandas as pd
import requests
from supabase import create_client, Client
from dotenv import load_dotenv


#configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("etl/etl_pipeline.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

#cargar variables de entorno
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")


#esquema esperado de columnas para validación
COLUMNAS_REQUERIDAS = [
    "fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar",
    "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide",
    "density", "ph", "sulphates", "alcohol", "quality"
]

COLUMNAS_SUPABASE = COLUMNAS_REQUERIDAS + ["categoria"]


def validar_esquema(df: pd.DataFrame, columnas: list, fuente: str) -> bool:
    """Valida que el DataFrame contenga las columnas requeridas."""
    faltantes = [c for c in columnas if c not in df.columns]
    if faltantes:
        logger.error(f"[{fuente}] Columnas faltantes: {faltantes}")
        return False
    logger.info(f"[{fuente}] Esquema validado correctamente ({len(df)} filas).")
    return True



#fuente 1 csv local
def extraer_csv(ruta: str = "data/winequality_clean.csv") -> pd.DataFrame:
    """Carga y normaliza el CSV original de vinos."""
    try:
        df = pd.read_csv(ruta, sep=",")
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        logger.info(f"[CSV] Archivo cargado: {ruta} ({len(df)} filas)")
        if not validar_esquema(df, COLUMNAS_REQUERIDAS, "CSV"):
            return pd.DataFrame()
        return df
    except FileNotFoundError as e:
        logger.error(f"[CSV] Archivo no encontrado en {ruta}: {e}")
        return pd.DataFrame()
    except pd.errors.ParserError as e:
        logger.error(f"[CSV] Error de formato/parseo al leer el CSV {ruta}: {e}")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"[CSV] Error inesperado al procesar {ruta}: {type(e).__name__} - {e}")
        return pd.DataFrame()



#fuente 2 api
def extraer_api(df_csv: pd.DataFrame, muestra: int = 5) -> pd.DataFrame:
    """Envia filas del CSV a POST /predict y retorna predicciones."""
    resultados = []
    if df_csv.empty:
        logger.warning("[API] DataFrame CSV vacio, se omite extraccion API.")
        return pd.DataFrame()

    filas = df_csv.head(muestra).to_dict(orient="records")
    FEATURES = [c for c in COLUMNAS_REQUERIDAS if c != "quality"]

    for i, fila in enumerate(filas):
        payload = {k: fila[k] for k in FEATURES}
        payload["pH"] = payload.pop("ph")
        try:
            resp = requests.post(f"{API_URL}/predict", json=payload, timeout=5)
            resp.raise_for_status()
            data = resp.json()
            calidad = data.get("calidad_predicha")
            confianza = data.get("confianza_porcentaje")
            resultados.append({
                "fila": i,
                "calidad_predicha": calidad,
                "confianza_porcentaje": confianza,
                **payload
            })
            logger.info(f"[API] Fila {i} -> calidad_predicha: {calidad} ({confianza}%)")
        except requests.exceptions.ConnectionError as e:
            logger.error(f"[API] Error de conexión a {API_URL} en fila {i}: {e}. Deteniendo reintentos.")
            break
        except requests.exceptions.HTTPError as e:
            logger.error(f"[API] Error HTTP al consultar fila {i}: {e}")
        except requests.exceptions.Timeout as e:
            logger.error(f"[API] Timeout al consultar fila {i}: {e}")
        except Exception as e:
            logger.error(f"[API] Error inesperado en fila {i}: {type(e).__name__} - {e}")

    df_api = pd.DataFrame(resultados)
    if not df_api.empty:
        logger.info(f"[API] {len(df_api)} predicciones obtenidas.")
    return df_api



#fuente 3 supabase
def extraer_supabase() -> pd.DataFrame:
    """Extrae todos los registros de la tabla 'vinos' en Supabase."""
    try:
        if not SUPABASE_URL or not SUPABASE_KEY:
            logger.error("[Supabase] Variables de entorno no configuradas.")
            return pd.DataFrame()

        client: Client = create_client(SUPABASE_URL, SUPABASE_KEY)



        todos_los_datos = []
        offset = 0
        LOTE = 1000
        while True:
            respuesta = client.table("vinos").select("*").range(offset, offset + LOTE - 1).execute()
            lote_datos = respuesta.data
            if not lote_datos:
                break
            todos_los_datos.extend(lote_datos)
            if len(lote_datos) < LOTE:
                break
            offset += LOTE
        
        
        datos = todos_los_datos

        if not datos:
            logger.warning("[Supabase] Tabla 'vinos' vacia o sin datos.")
            return pd.DataFrame()

        df = pd.DataFrame(datos)
        logger.info(f"[Supabase] {len(df)} filas extraidas de tabla 'vinos'.")

        if not validar_esquema(df, COLUMNAS_SUPABASE, "Supabase"):
            return pd.DataFrame()

        return df

    except Exception as e:
        logger.error(f"[Supabase] Error en extracción de Supabase ({type(e).__name__}): {e}")
        return pd.DataFrame()



#función de transformación
def transformar(df_csv: pd.DataFrame, df_supabase: pd.DataFrame) -> pd.DataFrame:
    """Consolida CSV y Supabase eliminando duplicados."""
    try:
        if "categoria" not in df_csv.columns:
            df_csv = df_csv.copy()
            df_csv["categoria"] = None

        df_combined = pd.concat([df_csv, df_supabase], ignore_index=True)
        antes = len(df_combined)
        df_combined = df_combined.drop_duplicates(subset=COLUMNAS_REQUERIDAS)
        despues = len(df_combined)
        logger.info(f"[ETL] Consolidacion: {antes} filas -> {despues} tras deduplicacion.")
        return df_combined
    except KeyError as e:
        logger.error(f"[ETL] Error en transformación. Columna faltante: {e}")
        return pd.DataFrame()
    except TypeError as e:
        logger.error(f"[ETL] Error de tipos durante concatenación/deduplicación: {e}")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"[ETL] Error inesperado en transformacion ({type(e).__name__}): {e}")
        return pd.DataFrame()



#guardar reultdp
def guardar(df: pd.DataFrame, ruta: str = "data/winequality_integrado.csv") -> None:
    """Guarda el dataset integrado en CSV."""
    try:
        df.to_csv(ruta, index=False)
        logger.info(f"[ETL] Dataset integrado guardado en: {ruta} ({len(df)} filas)")
    except PermissionError as e:
        logger.error(f"[ETL] Error de permisos al escribir archivo en {ruta}: {e}")
    except Exception as e:
        logger.error(f"[ETL] Error inesperado al guardar en {ruta} ({type(e).__name__}): {e}")



#pipeline inicial
def run_pipeline():
    """Ejecuta el pipeline ETL completo integrando las 3 fuentes de datos."""
    logger.info("=" * 50)
    logger.info("INICIO DEL PIPELINE ETL")
    logger.info("=" * 50)

    df_csv = extraer_csv()
    df_api = extraer_api(df_csv, muestra=5)
    df_supabase = extraer_supabase()

    if not df_csv.empty and not df_supabase.empty:
        df_final = transformar(df_csv, df_supabase)
    elif not df_csv.empty:
        logger.warning("[ETL] Supabase vacio, usando solo CSV.")
        df_final = df_csv.copy()
        df_final["categoria"] = None
    else:
        logger.error("[ETL] Sin datos suficientes para continuar.")
        return

    guardar(df_final)

    if not df_api.empty:
        guardar(df_api, ruta="data/predicciones_api.csv")

    logger.info("=" * 50)
    logger.info("PIPELINE ETL COMPLETADO")
    logger.info(f"Fuente 1 (CSV):      {len(df_csv)} filas")
    logger.info(f"Fuente 2 (API):      {len(df_api)} predicciones")
    logger.info(f"Fuente 3 (Supabase): {len(df_supabase)} filas")
    logger.info(f"Dataset integrado:   {len(df_final)} filas")
    logger.info("=" * 50)


if __name__ == "__main__":
    run_pipeline()