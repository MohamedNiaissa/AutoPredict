import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OrdinalEncoder
import joblib

# Charger les données
df = pd.read_csv("cartest.csv")

# Préparer les données
df["fuel"].replace({'Diesel': 0, 'Petrol': 1, 'LPG': 2, 'CNG': 3, 'Electric': 4}, inplace=True)
df["transmission"].replace({'Automatic': 0, 'Manual': 1}, inplace=True)
df["brand"] = df["name"].str.split().str[0]
df = df.drop("name", axis=1)

# Séparer les features et la cible
y = df["selling_price"]
X = df.drop("selling_price", axis=1)

# Diviser les données
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Pipeline pour la préparation des données
cat_selector = ["fuel", "transmission", "brand"]
num_selector = ["year", "km_driven"]

tree_processor = make_column_transformer(
    (SimpleImputer(strategy="mean"), num_selector),
    (OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat_selector),
)

# Construire le modèle
rfr_pipeline = make_pipeline(tree_processor, RandomForestRegressor(n_estimators=100, random_state=42))
rfr_pipeline.fit(x_train, y_train)

# Sauvegarder le pipeline complet
joblib.dump(rfr_pipeline, "rfr_pipeline.pkl")
