import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
import joblib
import os

# 1. Cargar el dataset limpio
df = pd.read_csv("data/winequality_clean.csv")

# 2. Reagrupar calidades en 3 clases
def clasificar_calidad(q):
    if q <= 4:
        return "bajo"
    elif q <= 6:
        return "medio"
    else:
        return "premium"

df["categoria"] = df["quality"].apply(clasificar_calidad)

# 3. Separar features (X) y target (y)
X = df.drop(columns=["quality", "categoria"])
y = df["categoria"]

# 4. Dividir en entrenamiento y test (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. Aplicar SMOTE solo al set de entrenamiento
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

print("Distribución tras SMOTE:")
print(pd.Series(y_train_bal).value_counts())

# 6. Entrenar el modelo
modelo = RandomForestClassifier(n_estimators=200, random_state=42)
modelo.fit(X_train_bal, y_train_bal)

# 7. Evaluar con el test set real (sin SMOTE)
y_pred = modelo.predict(X_test)
print("\nResultados en test set real:")
print(classification_report(y_test, y_pred))

# 8. Guardar el modelo entrenado
os.makedirs("api", exist_ok=True)
joblib.dump(modelo, "api/wine_model.pkl")
print("\nModelo guardado en api/wine_model.pkl")

# 9. Guardar dataset balanceado
X_train_bal["categoria"] = y_train_bal.values
os.makedirs("data", exist_ok=True)
X_train_bal.to_csv("data/winequality_balanced.csv", index=False)
print("Dataset balanceado guardado en data/winequality_balanced.csv")
