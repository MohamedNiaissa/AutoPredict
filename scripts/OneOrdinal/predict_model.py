import mlflow.sklearn
import pandas as pd
from mlflow.artifacts import download_artifacts
import joblib

# Configurer MLflow
mlflow.set_tracking_uri("http://127.0.0.1:8080")

# Charger le modèle depuis MLflow
model_uri = "models:/RandomForestPipeline/7"
print("Chargement du modèle depuis MLflow...")
try:
    model = mlflow.sklearn.load_model(model_uri)
    print("Modèle chargé avec succès.")
except Exception as e:
    print(f"Erreur lors du chargement du modèle : {e}")
    raise

# Charger l'encodeur de `brand`
print("Chargement de l'encodeur pour 'brand'...")
try:
    brand_encoder_path = download_artifacts(
        artifact_path="brand_encoder.pkl",
        run_id="cc3bed0b8766495f991edbdf3a127578"
    )
    brand_encoder = joblib.load(brand_encoder_path)
    print("Encodeur chargé avec succès.")
except Exception as e:
    print(f"Erreur lors du chargement de l'encodeur : {e}")
    raise

# Mappings pour les colonnes catégoriques
FUEL_MAPPING = {"Diesel": 0, "Petrol": 1, "LPG": 2, "CNG": 3, "Electric": 4}
TRANSMISSION_MAPPING = {"Automatic": 0, "Manual": 1}

# Exemple de données utilisateur
user_data = {
    "year": 2015,
    "km_driven": 45000,
    "fuel": "Petrol",         # Texte, à mapper
    "transmission": "Manual",  # Texte, à mapper
    "brand": "Hyundai"        # Texte, à encoder
}

# Préparer les données pour la prédiction
print("Préparation des données...")
try:
    # Mapper `fuel` et `transmission`
    user_data["fuel"] = FUEL_MAPPING.get(user_data["fuel"])
    user_data["transmission"] = TRANSMISSION_MAPPING.get(user_data["transmission"])

    if user_data["fuel"] is None or user_data["transmission"] is None:
        raise ValueError(f"Valeurs invalides pour fuel ou transmission.")

    # Encoder `brand`
    if user_data["brand"] not in brand_encoder.categories_[0]:
        print(f"Brand '{user_data['brand']}' est inconnu. Il sera encodé en -1.")
        user_data["brand"] = -1  # Valeur par défaut pour catégories inconnues
    else:
        user_data["brand"] = brand_encoder.transform([[user_data["brand"]]])[0, 0]

except Exception as e:
    print(f"Erreur lors de la préparation des données : {e}")
    raise

# Convertir les données utilisateur en DataFrame
test_data = pd.DataFrame([user_data])

# Forcer les types corrects
test_data = test_data.astype({
    "year": "int64",
    "km_driven": "int64",
    "fuel": "int64",
    "transmission": "int64",
    "brand": "float64",  # Encodage spécifique pour gérer -1
})

print(f"Données prêtes pour la prédiction :\n{test_data}")

# Faire la prédiction
print("Prédiction en cours...")
try:
    predictions = model.predict(test_data)
    print(f"Prédiction : {predictions[0]}")
except Exception as e:
    print(f"Erreur lors de la prédiction : {e}")
    raise
