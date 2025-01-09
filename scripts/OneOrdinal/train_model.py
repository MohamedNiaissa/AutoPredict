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

# Remplacer les valeurs textuelles par des valeurs numériques
df["fuel"] = df["fuel"].replace({'Diesel': 0, 'Petrol': 1, 'LPG': 2, 'CNG': 3, 'Electric': 4})
df["transmission"] = df["transmission"].replace({'Automatic': 0, 'Manual': 1})
df["brand"] = df["name"].str.split().str[0]
df = df.drop("name", axis=1)

# Vérifier les valeurs manquantes
if df.isnull().any().any():
    print("Attention : Des valeurs manquantes sont présentes dans les données.")
    df.fillna(-1, inplace=True)

# Diviser les données
y = df["selling_price"]
X = df.drop("selling_price", axis=1)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Préparer le pipeline
cat_selector = ["fuel", "transmission", "brand"]
num_selector = ["year", "km_driven"]

tree_processor = make_column_transformer(
    (SimpleImputer(strategy="mean"), num_selector),
    (OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat_selector),
)

params = {"n_estimators": 100, "random_state": 42}
rfr_pipeline = make_pipeline(tree_processor, RandomForestRegressor(**params))

# Entraîner le modèle
print("Entraînement du modèle en cours...")
rfr_pipeline.fit(x_train, y_train)
print("Modèle entraîné avec succès.")

# Prédictions sur l'ensemble de test
y_pred = rfr_pipeline.predict(x_test)

# Sauvegarder un encodeur spécifique pour `brand`
try:
    ordinal_encoder = tree_processor.named_transformers_["ordinalencoder"]
    brand_categories = ordinal_encoder.categories_[2]  # `brand` est la 3ème colonne
    # Créer un encodeur spécifique pour `brand`
    brand_encoder = OrdinalEncoder(categories=[brand_categories], handle_unknown="use_encoded_value", unknown_value=-1)
    # Fitter l'encodeur pour qu'il soit prêt à l'emploi
    brand_encoder.fit(X[["brand"]])
    # Sauvegarder l'encodeur fitté
    joblib.dump(brand_encoder, "brand_encoder.pkl")
    print(f"Encodeur pour 'brand' sauvegardé avec succès : {brand_categories}")
except Exception as e:
    print(f"Erreur lors de la sauvegarde de l'encodeur : {e}")

# Configurer MLflow
mlflow.set_tracking_uri("http://127.0.0.1:8080")
mlflow.set_experiment("test_experiment")

# Loguer le modèle avec MLflow
with mlflow.start_run():
    # Loguer les paramètres
    mlflow.log_params(params)

    # Calculer et loguer les métriques
    r2_score = rfr_pipeline.score(x_test, y_test)
    print(f"Score R2 sur l'ensemble de test : {r2_score:.4f}")
    mlflow.log_metric("r2_score", r2_score)

    # Signature des données
    signature = infer_signature(x_test, y_pred)

    # Loguer l'encodeur
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
