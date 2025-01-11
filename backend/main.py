from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import mlflow.pyfunc
import logging
from fastapi.middleware.cors import CORSMiddleware
import shap
import matplotlib.pyplot as plt


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
MODEL_URI = "models:/OptimizedRandomForestModel/3"  # Mettre la version correcte de votre modèle
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
    
print(model.metadata.get_input_schema())


# Endpoint pour expliquer une prédiction
@app.post("/explain")
async def explain(features: CarFeatures):
    try:
        logger.info(f"Requête reçue pour explication : {features.dict()}")

        # Mapper les valeurs textuelles
        fuel = FUEL_MAPPING.get(features.fuel)
        transmission = TRANSMISSION_MAPPING.get(features.transmission)
        brand = BRAND_MAPPING.get(features.brand)
        owner = OWNER_MAPPING.get(features.owner)
        seller_type = SELLER_TYPE_MAPPING.get(features.seller_type)

        # Valider les entrées
        if fuel is None or transmission is None or brand is None or owner is None or seller_type is None:
            raise ValueError("Certaines valeurs des caractéristiques sont invalides.")

        # Préparer les données pour SHAP
        input_data = pd.DataFrame([{
            "year": features.year,
            "km_driven": features.km_driven,
            "fuel": fuel,
            "seller_type": seller_type,
            "transmission": transmission,
            "owner": owner,
            "brand": features.brand
        }])

        background = pd.DataFrame([{
            "year": 2010,
            "km_driven": 50000,
            "fuel": 1,
            "seller_type": 0,
            "transmission": 1,
            "owner": 0,
            "brand": "Hyundai"
        }])

        # Conversion des types
        expected_columns = ["year", "km_driven", "fuel", "seller_type", "transmission", "owner", "brand"]
        background = background.astype({
            "year": "int",
            "km_driven": "int",
            "fuel": "int",
            "seller_type": "int",
            "transmission": "int",
            "owner": "int",
            "brand": "str"
        })[expected_columns]

        input_data = input_data.astype({
            "year": "int",
            "km_driven": "int",
            "fuel": "int",
            "seller_type": "int",
            "transmission": "int",
            "owner": "int",
            "brand": "str"
        })[expected_columns]

        # Fonction prédictive alignée
        def predict_dataframe(data):
            data = pd.DataFrame(data, columns=expected_columns)
            return model.predict(data)

        # Expliquer avec KernelExplainer
        explainer = shap.KernelExplainer(predict_dataframe, background)
        shap_values = explainer.shap_values(input_data)

        # Base value
        base_value = explainer.expected_value

        # Prédiction totale
        prediction = base_value + sum(shap_values[0])

        # Générer des descriptions d'impacts
        def format_impact(impact, feature_name):
            if abs(impact) < 1e-6:
                return f"Pas d'impact significatif"
            elif impact > 0:
                return f"Contribution positive importante : +{impact:.2f}"
            else:
                return f"Réduction due à {feature_name} : {impact:.2f}"

        feature_impact = {
            col: format_impact(shap_values[0][i], col)
            for i, col in enumerate(input_data.columns)
        }

        # Formater la réponse
        response = {
            "base_value": round(base_value, 2),
            "prediction": round(prediction, 2),
            "feature_impact": feature_impact
        }

        logger.info("Explication générée avec succès.")
        return response

    except ValueError as e:
        logger.error(f"Erreur utilisateur : {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Erreur interne : {str(e)}")
        raise HTTPException(status_code=500, detail="Erreur interne du serveur.")




@app.post("/explain_visual")
async def explain_visual(features: CarFeatures):
    try:
        logger.info(f"Requête reçue pour visualisation : {features.dict()}")

        # Mapper les valeurs textuelles
        fuel = FUEL_MAPPING.get(features.fuel)
        transmission = TRANSMISSION_MAPPING.get(features.transmission)
        brand = BRAND_MAPPING.get(features.brand)
        owner = OWNER_MAPPING.get(features.owner)
        seller_type = SELLER_TYPE_MAPPING.get(features.seller_type)

        # Valider les entrées
        if fuel is None or transmission is None or brand is None or owner is None or seller_type is None:
            raise ValueError("Certaines valeurs des caractéristiques sont invalides.")

        # Préparer les données pour SHAP
        input_data = pd.DataFrame([{
            "year": features.year,
            "km_driven": features.km_driven,
            "fuel": fuel,
            "seller_type": seller_type,
            "transmission": transmission,
            "owner": owner,
            "brand": features.brand
        }])

        background = pd.DataFrame([{
            "year": 2010,
            "km_driven": 50000,
            "fuel": 1,
            "seller_type": 0,
            "transmission": 1,
            "owner": 0,
            "brand": "Hyundai"
        }])

        # Conversion des types
        expected_columns = ["year", "km_driven", "fuel", "seller_type", "transmission", "owner", "brand"]
        background = background.astype({
            "year": "int",
            "km_driven": "int",
            "fuel": "int",
            "seller_type": "int",
            "transmission": "int",
            "owner": "int",
            "brand": "str"
        })[expected_columns]

        input_data = input_data.astype({
            "year": "int",
            "km_driven": "int",
            "fuel": "int",
            "seller_type": "int",
            "transmission": "int",
            "owner": "int",
            "brand": "str"
        })[expected_columns]

        # Fonction prédictive alignée
        def predict_dataframe(data):
            data = pd.DataFrame(data, columns=expected_columns)
            return model.predict(data)

        # Expliquer avec KernelExplainer
        explainer = shap.KernelExplainer(predict_dataframe, background)
        shap_values = explainer.shap_values(input_data)

        # # Visualiser avec force_plot
        # shap.force_plot(
        #     explainer.expected_value,
        #     shap_values[0],
        #     input_data.iloc[0],
        #     matplotlib=True
        # )

         
        # # Sauvegarder l'image
        # plt.savefig("shap_force_plot.png")
        # plt.close()

        shap.waterfall_plot(shap.Explanation(values=shap_values[0], base_values=explainer.expected_value, data=input_data.iloc[0]))
        plt.savefig("shap_waterfall_plot.png")
        plt.close()
       

        logger.info("Visualisation générée et sauvegardée avec succès.")
        return {"message": "Visualisation générée et sauvegardée sous shap_force_plot.png"}

    except ValueError as e:
        logger.error(f"Erreur utilisateur : {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Erreur interne : {str(e)}")
        raise HTTPException(status_code=500, detail="Erreur interne du serveur.")
