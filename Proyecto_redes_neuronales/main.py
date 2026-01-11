import pandas as pd
import sqlite3
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# --- CONFIGURACIÓN ---
csv_file_path = 'C:\\Users\\aldok\\Documents\\Proyecto_redes_neuronales\\Churn_Modelling.csv'
db_name = 'banco.db'
table_name = 'churn_data'

print("--- PASO 1: CARGA DE DATOS ---")
print("Cargando archivo CSV...")
df_original = pd.read_csv(csv_file_path)
conn = sqlite3.connect(db_name)

print(f"Guardando datos en la tabla '{table_name}'...")
df_original.to_sql(table_name, conn, if_exists='replace', index=False)

print("\n--- PASO 2: LIMPIEZA SQL ---")
query = f"""
SELECT
    CreditScore,
    Age,
    Tenure,
    Balance,
    NumOfProducts,
    HasCrCard,
    IsActiveMember,
    EstimatedSalary,
    -- Encoding de Género (Binario)
    CASE WHEN Gender = 'Male' THEN 1 ELSE 0 END as Gender_Male,
    -- Traemos Geography tal cual para usar LabelEncoder en Python
    Geography,
    -- Target
    Exited
FROM {table_name}
"""

df_clean = pd.read_sql_query(query, conn)
conn.close()

# APLICAR LABEL ENCODER 
print("Aplicando Label Encoder a Geografía...")
le = LabelEncoder()
df_clean['Geography'] = le.fit_transform(df_clean['Geography'])
print("Mapeo de Países:", dict(zip(le.classes_, le.transform(le.classes_))))

#SPLIT & SCALING
print("\n--- PASO 3: PREPARACIÓN ---")
X = df_clean.drop('Exited', axis=1)
y = df_clean['Exited']

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

#Escalado
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Datos de entrenamiento: {X_train_scaled.shape}")
print(f"Datos de prueba: {X_test_scaled.shape}")

#REGRESIÓN LOGÍSTICA MÚLTIPLE
print("\n--- PASO 4: ENTRENAMIENTO DEL MODELO ---")

model = LogisticRegression(class_weight='balanced', random_state=42)

model.fit(X_train_scaled, y_train)
print("¡Modelo entrenado exitosamente!")

#  EVALUACIÓN Y RESULTADOS 
print("\n--- PASO 5: RESULTADOS ---")
y_pred = model.predict(X_test_scaled)

print("Reporte de Clasificación:")
print(classification_report(y_test, y_pred))

print("\nMatriz de Confusión (Verdaderos vs Predichos):")
print(confusion_matrix(y_test, y_pred))

#INTERPRETACIÓN DE LA ECUACIÓN 
print("\n--- VARIABLES MÁS IMPORTANTES (Coeficientes) ---")
coeficientes = pd.DataFrame(model.coef_[0], index=X.columns, columns=['Peso (Importancia)'])
coeficientes['Impacto_Absoluto'] = coeficientes['Peso (Importancia)'].abs()
print(coeficientes.sort_values(by='Impacto_Absoluto', ascending=False).drop('Impacto_Absoluto', axis=1))

print("\n INTERPRETACIÓN:")
print(" - Peso Positivo (+): Aumenta el riesgo de fuga.")
print(" - Peso Negativo (-): Disminuye el riesgo (protege al cliente).")