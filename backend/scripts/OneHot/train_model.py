import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error

# Charger les données
df = pd.read_csv("./data/cartest.csv")

# Préparer les colonnes
df["brand"] = df["name"].str.split().str[0]
df = df.drop("name", axis=1)

# Diviser les données
X = df.drop("selling_price", axis=1)
y = df["selling_price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Définir les colonnes
categorical_columns = ["fuel", "transmission", "brand"]
numerical_columns = ["year", "km_driven"]

# Préparer les transformateurs
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])
numerical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean"))
])

# Combiner les transformateurs
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, numerical_columns),
        ("cat", categorical_transformer, categorical_columns)
    ]
)

# Créer le pipeline
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))
])

# Entraîner le modèle
print("Entraînement en cours...")
model.fit(X_train, y_train)
print("Modèle entraîné avec succès.")

# Évaluer le modèle
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error : {mse:.2f}")

# Configurer MLflow
mlflow.set_tracking_uri("http://127.0.0.1:8080")
mlflow.set_experiment("simple_model_experiment")

# # Ajouter l'exemple d'entrée
# input_example = pd.DataFrame({
#     "year": [2015],
#     "km_driven": [45000],
#     "fuel": ["Petrol"],
#     "transmission": ["Manual"],
#     "brand": ["Hyundai"]
# })

# Signature des données
from mlflow.models.signature import infer_signature
signature = infer_signature(X_train, model.predict(X_train))

# Loguer le modèle avec l'exemple d'entrée et la signature
with mlflow.start_run():
    mlflow.log_metric("mse", mse)
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        registered_model_name="SimpleRandomForestPipeline",
        signature=signature,
        # input_example=input_example
    )
    print("Modèle logué avec succès dans MLflow avec signature et exemple d'entrée.")
