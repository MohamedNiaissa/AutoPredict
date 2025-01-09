from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI()

# Charger le pipeline
try:
    pipeline = joblib.load("rfr_pipeline.pkl")  # Charger le pipeline
except Exception as e:
    raise RuntimeError("Erreur lors du chargement du pipeline. Vérifiez le fichier .pkl.") from e

class CarFeatures(BaseModel):
    year: int
    km_driven: int
    fuel: str
    transmission: str
    brand: str

# Dictionnaires de mappage pour les valeurs textuelles
FUEL_MAPPING = {"Diesel": 0, "Petrol": 1, "LPG": 2, "CNG": 3, "Electric": 4}
TRANSMISSION_MAPPING = {"Automatic": 0, "Manual": 1}

@app.get("/status")
async def get_status():
    return {"status": "Model is ready for prediction"}

@app.post("/predict")
async def predict(features: CarFeatures):
    try:
        # Mapper les valeurs textuelles
        fuel = FUEL_MAPPING.get(features.fuel)
        transmission = TRANSMISSION_MAPPING.get(features.transmission)

        if fuel is None or transmission is None:
            raise ValueError("Valeur de fuel ou transmission invalide.")

        # Préparer les données pour la prédiction
        input_data = pd.DataFrame([{
            "year": features.year,
            "km_driven": features.km_driven,
            "fuel": fuel,
            "transmission": transmission,
            "brand": features.brand,  # Encodé automatiquement par le pipeline
        }])

        # Faire une prédiction avec le pipeline
        prediction = pipeline.predict(input_data)[0]
        return {"predicted_selling_price": round(prediction, 2)}

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Erreur interne du serveur.")
