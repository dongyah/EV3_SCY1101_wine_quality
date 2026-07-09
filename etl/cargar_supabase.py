import pandas as pd
from supabase import create_client
from dotenv import load_dotenv
import os
import logging
import sys

#logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("etl/carga_supabase.log", encoding="utf-8")
    ]
)
logger = logging.getLogger(__name__)



#cargar variables de entorno
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    logger.error("Faltan variables de entorno SUPABASE_URL o SUPABASE_KEY en el archivo .env")
    sys.exit(1)



#conexion a supabase
try:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    logger.info("Conexión a Supabase exitosa")
except Exception as e:
    logger.error(f"No se pudo inicializar el cliente de Supabase ({type(e).__name__}): {e}")
    sys.exit(1)


## Cargar CSV y validar esquema
RUTA_CSV = "data/winequality_clean.csv"

COLUMNAS_ESPERADAS = {
    "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
    "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
    "pH", "sulphates", "alcohol", "quality"   
}

try:
    df = pd.read_csv(RUTA_CSV)
    logger.info(f"CSV cargado: {len(df)} filas, {len(df.columns)} columnas")
except FileNotFoundError as e:
    logger.error(f"No se encontró el archivo {RUTA_CSV}: {e}")
    sys.exit(1)
except pd.errors.ParserError as e:
    logger.error(f"Error de formato al procesar el archivo CSV {RUTA_CSV}: {e}")
    sys.exit(1)
except Exception as e:
    logger.error(f"Error inesperado al leer CSV en {RUTA_CSV} ({type(e).__name__}): {e}")
    sys.exit(1)

# Validación de esquema
columnas_presentes = set(df.columns)

columnas_faltantes = COLUMNAS_ESPERADAS - columnas_presentes

if columnas_faltantes:
    logger.error(f"Columnas faltantes en el CSV: {columnas_faltantes}")
    sys.exit(1)
logger.info("Validación de esquema: OK — todas las columnas requeridas están presentes")

# Validar tipos numéricos
columnas_numericas = [c for c in df.columns if c != "quality"]
if not all(pd.api.types.is_numeric_dtype(df[c]) for c in columnas_numericas):
    logger.warning("Algunas columnas tienen tipos inesperados, se intentará continuar")

# Validar nulos
nulos = df.isnull().sum().sum()
if nulos > 0:
    logger.warning(f"Se encontraron {nulos} valores nulos — se eliminarán")
    df = df.dropna()
    logger.info(f"Filas tras eliminar nulos: {len(df)}")
else:
    logger.info("Sin valores nulos — datos limpios")


#transformar columnas a minúsculas y reemplazar espacios por guiones bajos
df.columns = df.columns.str.replace(" ", "_").str.lower()
logger.info(f"Columnas normalizadas: {list(df.columns)}")



# Clasificar calidad en categorías
def clasificar_calidad(q):
    if q <= 4:
        return "bajo"
    elif q <= 6:
        return "medio"
    else:
        return "premium"

df["categoria"] = df["quality"].apply(clasificar_calidad)
logger.info(f"Distribución de categorías:\n{df['categoria'].value_counts().to_string()}")


# Limpiar tabla en Supabase antes de insertar
try:
    supabase.table("vinos").delete().neq("id", 0).execute()
    logger.info("Tabla 'vinos' limpiada exitosamente antes de la carga")
except Exception as e:
    logger.error(f"Error al limpiar la tabla 'vinos' en Supabase ({type(e).__name__}): {e}")
    sys.exit(1)



# Cargar datos a Supabase en lotes
registros = df.to_dict(orient="records")
LOTE = 100
total = len(registros)
insertados = 0
errores = 0

for i in range(0, total, LOTE):
    lote = registros[i:i + LOTE]
    try:
        supabase.table("vinos").insert(lote).execute()
        insertados += len(lote)
        logger.info(f"Progreso: {insertados}/{total} registros insertados")
    except Exception as e:
        errores += len(lote)
        logger.error(f"Error al insertar lote {i}–{i+LOTE} en Supabase ({type(e).__name__}): {e}")


# Finalizar carga
if errores == 0:
    logger.info(f"Carga completa: {insertados} registros cargados en Supabase sin errores")
else:
    logger.warning(f"Carga finalizada con ERRORES: {insertados} insertados, {errores} fallidos")
