import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import mean_squared_error, r2_score
from mlflow.models.signature import infer_signature
import mlflow
import mlflow.sklearn

# Charger les données
df = pd.read_csv("./data/cartest.csv")

# Préparer les données
df["fuel"] = df["fuel"].replace({'Diesel': 0, 'Petrol': 1, 'LPG': 2, 'CNG': 3, 'Electric': 4})
df["transmission"] = df["transmission"].replace({'Automatic': 0, 'Manual': 1})
df["owner"] = df["owner"].replace({'First Owner': 0, 'Second Owner': 1, 'Third Owner': 2, 'Fourth & Above Owner': 3, 'Test Drive Car': 4})
df["seller_type"] = df["seller_type"].replace({'Dealer': 0, 'Individual': 1, 'Trustmark Dealer': 2})
df["brand"] = df["name"].str.split().str[0]
df = df.drop("name", axis=1)

# Séparer les features et la cible
y = df["selling_price"]
X = df.drop("selling_price", axis=1)

# Diviser les données
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Pipeline pour la préparation des données
preprocessor = make_column_transformer(
    (SimpleImputer(strategy="mean"), ["year", "km_driven"]),
    (OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), ["brand"])
)

# Construire le pipeline
pipeline = make_pipeline(preprocessor, RandomForestRegressor(random_state=42))

# Paramètres pour RandomizedSearchCV
param_distributions = {
    "randomforestregressor__n_estimators": [100, 200, 300, 400, 500],
    "randomforestregressor__max_depth": [None, 10, 20, 30, 40],
    "randomforestregressor__min_samples_split": [2, 5, 10],
    "randomforestregressor__min_samples_leaf": [1, 2, 4],
    "randomforestregressor__max_features": ["auto", "sqrt", "log2"]
}

# RandomizedSearchCV
print("Recherche des meilleurs hyperparamètres en cours...")
search = RandomizedSearchCV(
    pipeline,
    param_distributions,
    n_iter=50,
    scoring="neg_mean_squared_error",
    cv=3,
    verbose=1,
    random_state=42,
    n_jobs=-1
)

search.fit(x_train, y_train)
best_model = search.best_estimator_
print(f"Meilleurs hyperparamètres : {search.best_params_}")

# Évaluation sur les données de test
y_pred = best_model.predict(x_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error : {mse}")
print(f"R² : {r2}")

# Signature pour MLflow
signature = infer_signature(x_test, y_pred)

# Intégration avec MLflow
mlflow.set_tracking_uri("http://127.0.0.1:8080")
mlflow.set_experiment("optimized_experiment")

with mlflow.start_run():
    mlflow.log_params(search.best_params_)
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("r2", r2)
    mlflow.sklearn.log_model(
        sk_model=best_model,
        artifact_path="optimized_model",
        signature=signature,
        registered_model_name="OptimizedRandomForestModel",
    )
    print("Modèle optimisé logué avec succès dans MLflow.")
