import pandas as pd
import mlflow.pyfunc

# Charger le modèle depuis MLflow
mlflow.set_tracking_uri("http://127.0.0.1:8080")
model_uri = "models:/OptimizedRandomForestModel/16"  # Remplacer par la bonne version
model = mlflow.pyfunc.load_model(model_uri)
print("Modèle chargé avec succès.")

# Exemple de données utilisateur
user_input = {
    "year": 2015,
    "km_driven": 45000,
    "fuel": 1,
    "transmission": 1,
    "owner": 0,
    "seller_type": 0,
    "brand": "Hyundai"  # Gardé en texte pour le modèle
}

# Convertir en DataFrame
test_data = pd.DataFrame([user_input])

# Afficher les colonnes fournies
print("Colonnes fournies :", test_data.columns.tolist())
print("Structure de test_data :")
print(test_data)

# Prédire
try:
    prediction = model.predict(test_data)
    print(f"Prédiction : {prediction[0]}")
except Exception as e:
    print(f"Erreur lors de la prédiction : {e}")
