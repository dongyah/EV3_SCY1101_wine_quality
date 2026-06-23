"""
Dashboard interactivo - Calidad de Vino Tinto
Vistas: Ejecutiva, Técnica, Operativa
"""
import os
import sys
import joblib
import requests
import pandas as pd
import plotly.express as px
import dash
from dash import dcc, html, Input, Output, State

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INTEGRADO_PATH = os.path.join(BASE_DIR, "data", "winequality_integrado.csv")
CLEAN_PATH = os.path.join(BASE_DIR, "data", "winequality_clean.csv")
MODEL_PATH = os.path.join(BASE_DIR, "api", "wine_model.pkl")
API_URL = "http://127.0.0.1:8000/predict"

COLOR_CATEGORIA = {"bajo": "#C1502E", "medio": "#B08D2B", "premium": "#1F6F50"}

CAMPOS = ["fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar",
          "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide", "density",
          "ph", "sulphates", "alcohol"]

LABELS = {
    "fixed_acidity": "Acidez fija", "volatile_acidity": "Acidez volátil",
    "citric_acid": "Ácido cítrico", "residual_sugar": "Azúcar residual",
    "chlorides": "Cloruros", "free_sulfur_dioxide": "SO2 libre",
    "total_sulfur_dioxide": "SO2 total", "density": "Densidad",
    "ph": "pH", "sulphates": "Sulfatos", "alcohol": "Alcohol (%)"
}

def clasificar_calidad(q):
    if q <= 4:
        return "bajo"
    elif q <= 6:
        return "medio"
    else:
        return "premium"

def estilizar(fig):
    """Aplica tipografía y tema consistentes a cualquier gráfico Plotly."""
    fig.update_layout(
        template="plotly_white",
        font=dict(family="Montserrat, sans-serif", size=13, color="#232323"),
        title_font=dict(family="Poppins, sans-serif", size=16, color="#232323"),
        margin=dict(t=50, l=40, r=20, b=40),
    )
    return fig

try:
    df = pd.read_csv(INTEGRADO_PATH)
except FileNotFoundError:
    print(f"ERROR: no se encontró {INTEGRADO_PATH}. Ejecuta primero etl/main_etl.py")
    sys.exit(1)

df["categoria"] = df["quality"].apply(clasificar_calidad)

try:
    df_clean = pd.read_csv(CLEAN_PATH).dropna()
    df_clean["categoria"] = df_clean["quality"].apply(clasificar_calidad)
    X = df_clean.drop(columns=["quality", "categoria"])
    y = df_clean["categoria"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    modelo = joblib.load(MODEL_PATH)
    y_pred = modelo.predict(X_test)
    ACCURACY = round(accuracy_score(y_test, y_pred) * 100, 1)
    CM = confusion_matrix(y_test, y_pred, labels=["bajo", "medio", "premium"])
    FEATURE_NAMES = X.columns.tolist()
    IMPORTANCIAS = modelo.feature_importances_
    MODELO_DISPONIBLE = True
except Exception as e:
    print(f"ADVERTENCIA: no se pudieron calcular métricas del modelo ({e})")
    MODELO_DISPONIBLE = False
    ACCURACY, CM, FEATURE_NAMES, IMPORTANCIAS = 0, None, [], []

app = dash.Dash(__name__)
app.title = "Calidad de Vino - Dashboard"
app.config.suppress_callback_exceptions = True

TAB_STYLE = {
    "fontFamily": "Montserrat, sans-serif", "fontWeight": "500",
    "padding": "14px", "border": "none",
    "borderBottom": "3px solid transparent", "color": "#6B6B6B",
}
TAB_SELECTED_STYLE = {
    "fontFamily": "Poppins, sans-serif", "fontWeight": "600",
    "padding": "14px", "border": "none",
    "borderBottom": "3px solid #7B1E3D", "color": "#7B1E3D",
    "backgroundColor": "#FFFFFF",
}

app.layout = html.Div([
    html.H1("Calidad de Vino Tinto", className="dashboard-header"),
    html.P("Pipeline ETL + SMOTE + RandomForest — 3 fuentes integradas",
           className="dashboard-subtitle"),
    dcc.Tabs(id="tabs", value="tab-exec", children=[
        dcc.Tab(label="Ejecutiva", value="tab-exec", style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
        dcc.Tab(label="Técnica", value="tab-tech", style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
        dcc.Tab(label="Operativa", value="tab-ops", style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
    ]),
    html.Div(id="tabs-content")
])



def layout_ejecutiva():
    total = len(df)
    pct_premium = round((df["categoria"] == "premium").mean() * 100, 1)
    pct_medio = round((df["categoria"] == "medio").mean() * 100, 1)

    fig_dist = estilizar(px.pie(
        df, names="categoria", color="categoria",
        color_discrete_map=COLOR_CATEGORIA,
        title="Distribución del catálogo por categoría de calidad"
    ))

    def kpi(valor, etiqueta):
        return html.Div([
            html.P(valor, className="kpi-value"),
            html.P(etiqueta, className="kpi-label"),
        ], className="kpi-card")

    return html.Div([
        html.Div([
            kpi(f"{total}", "Vinos analizados"),
            kpi(f"{pct_premium}%", "Calidad premium"),
            kpi(f"{pct_medio}%", "Calidad media"),
            kpi(f"{ACCURACY}%", "Precisión del modelo"),
        ], className="kpi-row"),
        html.Div(dcc.Graph(figure=fig_dist), className="chart-card")
    ])



def layout_tecnica():
    if not MODELO_DISPONIBLE:
        return html.Div("No se pudo cargar el modelo o las métricas.", style={"padding": "20px"})

    fig_imp = estilizar(px.bar(
        x=FEATURE_NAMES, y=IMPORTANCIAS,
        labels={"x": "Variable", "y": "Importancia"},
        title="Importancia de variables en el modelo (RandomForest)",
        color=IMPORTANCIAS, color_continuous_scale="Viridis"
    ))

    fig_cm = estilizar(px.imshow(
        CM, x=["bajo", "medio", "premium"], y=["bajo", "medio", "premium"],
        text_auto=True, color_continuous_scale="Blues",
        labels={"x": "Predicho", "y": "Real", "color": "Cantidad"},
        title=f"Matriz de confusión (test set real, accuracy: {ACCURACY}%)"
    ))

    corr = df[CAMPOS].corr()
    fig_corr = estilizar(px.imshow(
        corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
        title="Correlación entre variables fisicoquímicas"
    ))

    return html.Div([
        html.Div(dcc.Graph(figure=fig_imp), className="chart-card"),
        html.Div(dcc.Graph(figure=fig_cm), className="chart-card"),
        html.Div(dcc.Graph(figure=fig_corr), className="chart-card"),
    ])



def layout_operativa():
    return html.Div([
        html.H3("Filtros analíticos", className="section-title"),
        html.Div([
            html.Label("Rango de alcohol (%)", className="input-field"),
            dcc.RangeSlider(
                id="filtro-alcohol",
                min=float(df["alcohol"].min()), max=float(df["alcohol"].max()),
                value=[float(df["alcohol"].min()), float(df["alcohol"].max())],
                tooltip={"placement": "bottom", "always_visible": True}
            ),
        ], style={"padding": "0 24px"}),
        html.Div(dcc.Graph(id="grafico-filtrado"), className="chart-card"),

        html.H3("Predicción en vivo (consume API FastAPI /predict)", className="section-title"),
        html.Div([
            html.Div([
                html.Label(LABELS[f]),
                dcc.Input(id=f"input-{f}", type="number",
                          value=round(float(df[f].mean()), 2), style={"width": "100%"})
            ], className="input-field",
               style={"width": "22%", "display": "inline-block", "padding": "8px"})
            for f in CAMPOS
        ], style={"padding": "0 16px"}),
        html.Div(
            html.Button("Predecir Calidad", id="boton-predecir", n_clicks=0, className="predict-button"),
            style={"padding": "8px 24px"}
        ),
        html.Div(id="resultado-prediccion", style={"padding": "8px 24px", "fontSize": "16px"})
    ])



@app.callback(Output("tabs-content", "children"), Input("tabs", "value"))
def render_tab(tab):
    if tab == "tab-exec":
        return layout_ejecutiva()
    elif tab == "tab-tech":
        return layout_tecnica()
    elif tab == "tab-ops":
        return layout_operativa()



@app.callback(Output("grafico-filtrado", "figure"), Input("filtro-alcohol", "value"))
def actualizar_filtro(rango_alcohol):
    dff = df[(df["alcohol"] >= rango_alcohol[0]) & (df["alcohol"] <= rango_alcohol[1])]
    fig = px.scatter(
        dff, x="alcohol", y="volatile_acidity", color="categoria",
        color_discrete_map=COLOR_CATEGORIA,
        title=f"Vinos filtrados: {len(dff)} de {len(df)}"
    )
    return estilizar(fig)



@app.callback(
    Output("resultado-prediccion", "children"),
    Input("boton-predecir", "n_clicks"),
    [State(f"input-{c}", "value") for c in CAMPOS]
)
def predecir(n_clicks, *valores):
    if n_clicks == 0:
        return ""
    payload = dict(zip(CAMPOS, valores))
    payload["pH"] = payload.pop("ph")
    try:
        r = requests.post(API_URL, json=payload, timeout=5)
        r.raise_for_status()
        data = r.json()
        return html.Div([
            html.Strong(f"Calidad predicha: {data['calidad_predicha']}"),
            html.P(f"Confianza: {data['confianza_porcentaje']}%")
        ], className="predict-result-success")
    except requests.exceptions.ConnectionError:
        return html.Div("Error: la API no está corriendo. Ejecuta: python -m uvicorn api.main:app --reload",
                         className="predict-result-error")
    except Exception as e:
        return html.Div(f"Error en la predicción: {e}", className="predict-result-error")

if __name__ == "__main__":
    app.run(debug=True, port=8050)