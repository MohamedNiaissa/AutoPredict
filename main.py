from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import mlflow.pyfunc
import logging

# Configurer les logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Configurer l'URI de MLflow et charger le modèle
mlflow.set_tracking_uri("http://127.0.0.1:8080")
MODEL_URI = "models:/SimpleRandomForestPipeline/2"  # Remplacez par votre modèle
try:
    model = mlflow.pyfunc.load_model(MODEL_URI)
    logger.info("Modèle chargé avec succès depuis MLflow")
except Exception as e:
    raise RuntimeError(f"Erreur lors du chargement du modèle : {e}")

# Mappings pour `fuel`, `transmission` et `brand`
FUEL_MAPPING = {"Diesel": 0, "Petrol": 1, "LPG": 2, "CNG": 3, "Electric": 4}
TRANSMISSION_MAPPING = {"Automatic": 0, "Manual": 1}
BRAND_MAPPING = {  # Remplacez par les catégories exactes de votre dataset
    "Hyundai": 0,
    "Maruti": 1,
    "Ford": 2,
    "Toyota": 3,
    # Ajoutez toutes les marques utilisées dans vos données d'entraînement
}

class CarFeatures(BaseModel):
    year: int
    km_driven: int
    fuel: str
    transmission: str
    brand: str

@app.get("/status")
async def get_status():
    return {"status": "Model is ready for prediction"}

@app.get("/metadata")
async def get_metadata():
    return {
        "model_uri": MODEL_URI,
        "fuel_mapping": FUEL_MAPPING,
        "transmission_mapping": TRANSMISSION_MAPPING,
        "brand_mapping": list(BRAND_MAPPING.keys()),
        "status": "Modèle chargé avec succès"
    }

@app.post("/predict")
async def predict(features: CarFeatures):
    try:
        logger.info(f"Requête reçue : {features.dict()}")

        # Mapper les valeurs textuelles
        fuel = FUEL_MAPPING.get(features.fuel)
        transmission = TRANSMISSION_MAPPING.get(features.transmission)
        brand = BRAND_MAPPING.get(features.brand)

        # Valider les entrées
        if fuel is None:
            raise ValueError(f"Fuel invalide : {features.fuel}. Valeurs possibles : {list(FUEL_MAPPING.keys())}")
        if transmission is None:
            raise ValueError(f"Transmission invalide : {features.transmission}. Valeurs possibles : {list(TRANSMISSION_MAPPING.keys())}")
        if brand is None:
            raise ValueError(f"Brand invalide : {features.brand}. Valeurs possibles : {list(BRAND_MAPPING.keys())}")

        # Préparer les données
        input_data = pd.DataFrame([{
            "year": features.year,
            "km_driven": features.km_driven,
            "fuel": fuel,
            "transmission": transmission,
            "brand": brand,
        }])

        logger.info(f"Données préparées pour le modèle : {input_data}")

        # Faire une prédiction
        prediction = model.predict(input_data)
        logger.info(f"Prédiction effectuée : {prediction[0]}")

        return {"predicted_selling_price": round(prediction[0], 2)}

    except ValueError as e:
        logger.error(f"Erreur utilisateur : {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Erreur interne : {str(e)}")
        raise HTTPException(status_code=500, detail="Erreur interne du serveur.")
