import pandas as pd
import mlflow.pyfunc
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Charger le modèle depuis MLflow
mlflow.set_tracking_uri("http://127.0.0.1:8080")
model_uri = "models:/OptimizedRandomForestModel/3"  # Mettre la bonne version du modèle
model = mlflow.pyfunc.load_model(model_uri)
print("Modèle chargé avec succès.")

# Charger les données de test
df = pd.read_csv("./data/cartest.csv")

# Préparer les données comme dans le script d'entraînement
df["fuel"].replace({'Diesel': 0, 'Petrol': 1, 'LPG': 2, 'CNG': 3, 'Electric': 4}, inplace=True)
df["transmission"].replace({'Automatic': 0, 'Manual': 1}, inplace=True)
df["owner"].replace({'First Owner': 0, 'Second Owner': 1, 'Third Owner': 2, 'Fourth & Above Owner': 3, 'Test Drive Car': 4}, inplace=True)
df["seller_type"].replace({'Dealer': 0, 'Individual': 1, 'Trustmark Dealer': 2}, inplace=True)
df["brand"] = df["name"].str.split().str[0]
df = df.drop("name", axis=1)

# Séparer les features et la cible
y = df["selling_price"]
X = df.drop("selling_price", axis=1)

# Diviser les données (utiliser les données de test pour l'affichage)
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Faire des prédictions
predictions = model.predict(X_test)

# Créer un DataFrame pour afficher les résultats
result = X_test.copy()
result["actual_price"] = y_test.values
result["predicted_price"] = predictions
result["difference"] = abs(result["actual_price"] - result["predicted_price"])

# Formater les colonnes en nombres entiers pour la lisibilité
result["actual_price"] = result["actual_price"].astype(int)
result["predicted_price"] = result["predicted_price"].round(0).astype(int)
result["difference"] = result["difference"].round(0).astype(int)

# Afficher un échantillon de résultats
result_sample = result.sort_values(by="difference", ascending=False).head(10)
print("Comparaison des prix réels et prédits :")
print(result_sample)



# Indices pour les marques
indices = range(len(result_sample))

# Création du graphique
plt.figure(figsize=(14, 8))

# Courbe pour les prix réels
plt.plot(indices, result_sample["actual_price"], label="Prix Réel", marker="o", color="blue", linewidth=2)

# Courbe pour les prix prédits
plt.plot(indices, result_sample["predicted_price"], label="Prix Prédit", marker="o", color="orange", linewidth=2)

# Zone colorée pour la différence
plt.fill_between(
    indices,
    result_sample["actual_price"],
    result_sample["predicted_price"],
    color="gray",
    alpha=0.3,
    label="Différence (abs)"
)

# Ajout des labels
plt.xticks(indices, result_sample["brand"], rotation=45)
plt.title("Comparaison des prix réels et prédits avec différence", fontsize=14)
plt.ylabel("Prix", fontsize=12)
plt.xlabel("Marque", fontsize=12)
plt.legend(fontsize=12)
plt.grid(axis="y", linestyle="--", alpha=0.7)

plt.tight_layout()
plt.show()



# Sauvegarder les résultats dans un fichier CSV pour analyse ultérieure
#result_sample.to_csv("evaluation_results.csv", index=False)
#print("Les résultats ont été sauvegardés dans 'evaluation_results.csv'.")
