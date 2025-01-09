from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import mlflow.pyfunc #Pyfunc permet de charger le modèle MLflow
from mlflow.artifacts import download_artifacts
import logging
import joblib

# Configurer les logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Configurer l'URI de MLflow et charger le modèle
mlflow.set_tracking_uri("http://127.0.0.1:8080")
MODEL_URI = "models:/RandomForestPipeline/1"
try:
    model = mlflow.pyfunc.load_model(MODEL_URI)
    logger.info("Modèle chargé avec succès depuis MLflow")
except Exception as e:
    raise RuntimeError(f"Erreur lors du chargement du modèle : {e}")

# Charger l'encodeur de `brand`
try:
    # Télécharger l'artefact brand_encoder.pkl depuis MLflow
    brand_encoder_path = download_artifacts(artifact_path="brand_encoder.pkl")
    # Charger l'encodeur avec joblib
    brand_encoder = joblib.load(brand_encoder_path)
    logger.info("Encoder pour `brand` chargé avec succès")
except Exception as e:
    raise RuntimeError(f"Erreur lors du chargement de l'encoder : {e}")

class CarFeatures(BaseModel):
    year: int
    km_driven: int
    fuel: str
    transmission: str
    brand: str

FUEL_MAPPING = {"Diesel": 0, "Petrol": 1, "LPG": 2, "CNG": 3, "Electric": 4}
TRANSMISSION_MAPPING = {"Automatic": 0, "Manual": 1}




@app.get("/status")
async def get_status():
    return {"status": "Model is ready for prediction"}

@app.get("/metadata")
async def get_metadata():
    return {
        "model_uri": MODEL_URI,
        "fuel_mapping": FUEL_MAPPING,
        "transmission_mapping": TRANSMISSION_MAPPING,
        "status": "Modèle chargé avec succès"
    }

@app.post("/predict")
async def predict(features: CarFeatures):
    try:
        logger.info(f"Requête reçue : {features.dict()}")

        fuel = FUEL_MAPPING.get(features.fuel)
        transmission = TRANSMISSION_MAPPING.get(features.transmission)

        if fuel is None:
            raise ValueError(f"Fuel invalide : {features.fuel}. Valeurs possibles : {list(FUEL_MAPPING.keys())}")
        if transmission is None:
            raise ValueError(f"Transmission invalide : {features.transmission}. Valeurs possibles : {list(TRANSMISSION_MAPPING.keys())}")

        # Encoder `brand` avec l'encodeur
        try:
            brand_encoded = brand_encoder.transform([[features.brand]])[0, 0]
        except Exception as e:
            raise ValueError(f"Brand invalide : {features.brand}. Erreur : {e}")


        input_data = pd.DataFrame([{
            "year": features.year,
            "km_driven": features.km_driven,
            "fuel": fuel,
            "transmission": transmission,
            "brand": brand_encoded,
        }])

        logger.info(f"Données préparées pour le modèle : {input_data}")

        prediction = model.predict(input_data)
        logger.info(f"Prédiction effectuée : {prediction[0]}")
        return {"predicted_selling_price": prediction[0]}

    except ValueError as e:
        logger.error(f"Erreur utilisateur : {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Erreur interne : {str(e)}")
        raise HTTPException(status_code=500, detail="Erreur interne du serveur.")
