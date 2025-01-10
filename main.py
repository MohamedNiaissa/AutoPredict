from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import mlflow.pyfunc
import logging
from fastapi.middleware.cors import CORSMiddleware

# Configurer les logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Créer une instance de l'application FastAPI
app = FastAPI()

# Configurer les paramètres CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permettre les requêtes provenant de tous les domaines
    allow_credentials=True,
    allow_methods=["*"],  # Permettre toutes les méthodes (GET, POST, etc.)
    allow_headers=["*"],  # Permettre tous les headers
)

# Configurer l'URI de MLflow et charger le modèle
mlflow.set_tracking_uri("http://127.0.0.1:8080")
MODEL_URI = "models:/RandomForestModel/16"  # Mettre la version correcte de votre modèle
try:
    model = mlflow.pyfunc.load_model(MODEL_URI)
    logger.info("Modèle chargé avec succès depuis MLflow")
except Exception as e:
    raise RuntimeError(f"Erreur lors du chargement du modèle : {e}")

# Mappings pour `fuel`, `transmission`, et `brand`
FUEL_MAPPING = {"Diesel": 0, "Petrol": 1, "LPG": 2, "CNG": 3, "Electric": 4}
TRANSMISSION_MAPPING = {"Automatic": 0, "Manual": 1}
BRAND_MAPPING = {  
    "Hyundai": 0,
    "Maruti": 1,
    "Ford": 2,
    "Toyota": 3,
}
OWNER_MAPPING = {
    "First Owner": 0,
    "Second Owner": 1,
    "Third Owner": 2,
    "Fourth & Above Owner": 3,
    "Test Drive Car": 4,
}
SELLER_TYPE_MAPPING = {
    "Dealer": 0,
    "Individual": 1,
    "Trustmark Dealer": 2,
}

# Définir le schéma d'entrée avec toutes les colonnes
class CarFeatures(BaseModel):
    year: int
    km_driven: int
    fuel: str
    transmission: str
    owner: str
    seller_type: str
    brand: str

# Endpoint pour vérifier le statut de l'API
@app.get("/status")
async def get_status():
    return {"status": "Model is ready for prediction"}

# Endpoint pour récupérer les métadonnées du modèle
@app.get("/metadata")
async def get_metadata():
    return {
        "model_uri": MODEL_URI,
        "fuel_mapping": FUEL_MAPPING,
        "transmission_mapping": TRANSMISSION_MAPPING,
        "brand_mapping": list(BRAND_MAPPING.keys()),
        "owner_mapping": list(OWNER_MAPPING.keys()),
        "seller_type_mapping": list(SELLER_TYPE_MAPPING.keys()),
        "status": "Modèle chargé avec succès"
    }

# Endpoint pour prédire le prix de vente d'une voiture
@app.post("/predict")
async def predict(features: CarFeatures):
    try:
        logger.info(f"Requête reçue : {features.dict()}")

        # Mapper les valeurs textuelles
        fuel = FUEL_MAPPING.get(features.fuel)
        transmission = TRANSMISSION_MAPPING.get(features.transmission)
        brand = BRAND_MAPPING.get(features.brand)
        owner = OWNER_MAPPING.get(features.owner)
        seller_type = SELLER_TYPE_MAPPING.get(features.seller_type)

        # Valider les entrées
        if fuel is None:
            raise ValueError(f"Fuel invalide : {features.fuel}. Valeurs possibles : {list(FUEL_MAPPING.keys())}")
        if transmission is None:
            raise ValueError(f"Transmission invalide : {features.transmission}. Valeurs possibles : {list(TRANSMISSION_MAPPING.keys())}")
        if brand is None:
            raise ValueError(f"Brand invalide : {features.brand}. Valeurs possibles : {list(BRAND_MAPPING.keys())}")
        if owner is None:
            raise ValueError(f"Owner invalide : {features.owner}. Valeurs possibles : {list(OWNER_MAPPING.keys())}")
        if seller_type is None:
            raise ValueError(f"Seller Type invalide : {features.seller_type}. Valeurs possibles : {list(SELLER_TYPE_MAPPING.keys())}")

        # Préparer les données
        input_data = pd.DataFrame([{
            "year": features.year,
            "km_driven": features.km_driven,
            "fuel": fuel,
            "transmission": transmission,
            "owner": owner,
            "seller_type": seller_type,
            "brand": features.brand,
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
