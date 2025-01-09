import mlflow.sklearn
import pandas as pd


mlflow.set_tracking_uri("http://127.0.0.1:8080")

# Charger un modèle depuis MLflow
model_uri = "models:/RandomForestPipeline/1"  
model = mlflow.sklearn.load_model(model_uri)

# Exemple de données de test
test_data = pd.DataFrame([{
    "year": 2015,
    "km_driven": 45000,
    "fuel": 0,
    "transmission": 1,
    "brand": "Hyundai"
}])

# Prédire
predictions = model.predict(test_data)
print(f"Prédiction : {predictions}")
