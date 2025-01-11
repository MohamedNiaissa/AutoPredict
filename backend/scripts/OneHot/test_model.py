import mlflow.sklearn
import pandas as pd

# Configurer MLflow
mlflow.set_tracking_uri("http://127.0.0.1:8080")

# Charger le modèle depuis MLflow
model_uri = "models:/SimpleRandomForestPipeline/2"
print("Chargement du modèle...")
model = mlflow.sklearn.load_model(model_uri)
print("Modèle chargé avec succès.")

# Exemple de données utilisateur
user_data = {
    "year": 2015,
    "km_driven": 45000,
    "fuel": "Petrol",
    "transmission": "Manual",
    "brand": "Hyundai"
}

# Convertir en DataFrame
input_data = pd.DataFrame([user_data])

# Faire une prédiction
print("Prédiction en cours...")
prediction = model.predict(input_data)
print(f"Prix estimé : {prediction[0]}")
