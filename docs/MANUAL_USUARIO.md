# Manual de Usuario — Dashboard de Calidad de Vino

## 1. Acceso
Abrir el navegador en http://127.0.0.1:8050
(requiere la API corriendo en http://127.0.0.1:8000)

## 2. Vistas disponibles
- **Vista Ejecutiva**: KPIs generales de calidad, distribución por categoría (bajo / medio / premium).
- **Vista Técnica**: métricas del modelo (accuracy, matriz de confusión, importancia de variables), train-test split.
- **Vista Operativa**: formulario de predicción individual a partir de variables fisicoquímicas.

## 3. Cómo hacer una predicción
1. Ir a la vista operativa.
2. Completar los campos solicitados (acidez, pH, alcohol, etc.).
3. Presionar "Predecir".
4. El sistema muestra la categoría predicha.

## 4. Interpretación de resultados
- **Bajo**: calidad ≤ 4
- **Medio**: calidad 5–6
- **Premium**: calidad ≥ 7

## 5. Soporte
Si el dashboard no responde o muestra error de conexión, verificar que la API esté corriendo en el puerto 8000 antes de reintentar.
