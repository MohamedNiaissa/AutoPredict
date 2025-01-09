import pandas as pd
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from mlflow.models.signature import infer_signature
import mlflow.sklearn
import joblib

# Charger les données
df = pd.read_csv("./data/cartest.csv")
df["fuel"].replace({'Diesel': 0, 'Petrol': 1, 'LPG': 2, 'CNG': 3, 'Electric': 4}, inplace=True)
df["transmission"].replace({'Automatic': 0, 'Manual': 1}, inplace=True)
df["brand"] = df["name"].str.split().str[0]
df = df.drop("name", axis=1)

# Division des données en ensembles d'entraînement et de test
y = df["selling_price"]
X = df.drop("selling_price", axis=1)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Configuration du pipeline de transformation et du modèle
cat_selector = ["fuel", "transmission", "brand"]
num_selector = ["year", "km_driven"]

# Préprocesseurs pour les colonnes catégoriques et numériques
tree_processor = make_column_transformer(
    (SimpleImputer(strategy="mean"), num_selector),
    (OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat_selector),
)

# Configuration du modèle Random Forest
params = {"n_estimators": 100, "random_state": 42}
rfr_pipeline = make_pipeline(tree_processor, RandomForestRegressor(**params))

# Entraînement du modèle
print("Entraînement du modèle en cours...")
rfr_pipeline.fit(x_train, y_train)
print("Modèle entraîné avec succès.")

# Prédictions sur l'ensemble de test
y_pred = rfr_pipeline.predict(x_test)

# Configuration de MLflow
mlflow.set_tracking_uri("http://127.0.0.1:8080")
mlflow.set_experiment("test_experiment")

# Sauvegarder l'encodeur pour `brand`
try:
    ordinal_encoder = tree_processor.named_transformers_["pipeline-1"]
    joblib.dump(ordinal_encoder, "brand_encoder.pkl")
    print("Encodeur pour 'brand' sauvegardé avec succès.")
except Exception as e:
    print(f"Erreur lors de la sauvegarde de l'encodeur : {e}")

# Loguer le modèle et les artefacts avec MLflow
with mlflow.start_run():
    # Loguer les paramètres du modèle
    mlflow.log_params(params)

    # Calculer et loguer les métriques
    r2_score = rfr_pipeline.score(x_test, y_test)
    print(f"Score R2 sur l'ensemble de test : {r2_score:.4f}")
    mlflow.log_metric("r2_score", r2_score)

    # Signature des données d'entrée et sortie
    signature = infer_signature(x_test, y_pred)

    # Loguer l'encodeur en tant qu'artefact
    mlflow.log_artifact("brand_encoder.pkl")

    # Loguer le modèle
    mlflow.sklearn.log_model(
        sk_model=rfr_pipeline,
        artifact_path="model",
        signature=signature,
        registered_model_name="RandomForestPipeline"
    )

    print("Modèle et artefacts logués avec succès dans MLflow.")

print("Run MLflow terminé avec succès.")
