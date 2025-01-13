# Projet : API FastAPI avec Modèle MLflow

Ce projet implémente une API REST avec **FastAPI** pour servir un modèle machine learning logué dans **MLflow**. L'utilisateur peut envoyer des caractéristiques de véhicules via une requête POST pour obtenir une estimation du prix de vente.

---

## Fonctionnalités

- **Endpoint **`/status`** : Permet de vérifier si l'API est prête.
- **Endpoint **`/predict`** : Permet d'envoyer les caractéristiques d'un véhicule et d'obtenir une estimation de son prix de vente.
- **Endpoint **`/metadata`** : Fournit des informations sur le modèle et les mappages utilisés pour la transformation des données.
- **Endpoint **`/explain`** : Permet de comprendre pourquoi le modèle a prédit une certaine valeur en affichant les impacts des caractéristiques sur la prédiction.
- **Endpoint **`/explain-visual`** : Fournit un graphique visuel basé sur SHAP pour illustrer les impacts des caractéristiques sur la prédiction.

---

## Prérequis

1. **Python 3.9+**
2. Installer les dépendances Python :
```bash
   cd backend
   ```
   ```bash
   pip install -r requirements.txt
   ```
3. **MLflow** doit être installé et le serveur MLflow doit être démarré.
   ```bash
   mlflow ui --host 127.0.0.1 --port 8080
   ```
4. Avoir un modèle logué dans MLflow avec un URI accessible (par exemple, `models:/RandomForestPipeline/1`).

---

## Organisation du Projet

```
/mon_projet
    |
    |-- main.py                # Code FastAPI pour servir l'API
    |-- requirements.txt       # Liste des dépendances
    |
    |-- data/                  # Données (exemple : cartest.csv)
    |-- mlruns/                # Dossier généré par MLflow (optionnel)
    |
    |-- scripts/OneOrdinal/    # Scripts liés à MLflow
        |-- train_model.py     # Script pour entraîner et enregistrer le modèle
        |-- predict_model.py   # Script pour effectuer des prédictions
        |-- model_evaluation.py # Script pour évaluer le modèle
```

---

## **Guide : Enregistrer et Tester un Modèle avec MLflow**

### **Étape 1 : Préparer les données**
Assurez-vous que les données nécessaires au modèle sont disponibles dans le dossier `data/` (exemple : `cartest.csv`).

---

### **Étape 2 : Entraîner et enregistrer le modèle**
1. Naviguez dans le dossier contenant les scripts :
   ```bash
   cd scripts/OneOrdinal
   ```

2. Lancez le script `train_model.py` pour entraîner et enregistrer le modèle dans MLflow :
   ```bash
   python train_model.py
   ```

3. Une fois l'exécution terminée :
   - Un nouveau modèle sera enregistré dans MLflow.
   - Les métriques d'entraînement, comme le **Mean Squared Error**, seront affichées.
   - Vous pouvez consulter le modèle et les artefacts associés dans l'interface MLflow :
     [http://127.0.0.1:8080](http://127.0.0.1:8080)

---

### **Étape 3 : Tester le modèle**
1. Lancez le script `predict_model.py` pour tester le modèle avec une observation personnalisée :
   ```bash
   python predict_model.py
   ```

2. Le script affichera les caractéristiques utilisateur, les données transformées et le prix prédit. Exemple :
   ```bash
   Données utilisateur : {'year': 2015, 'km_driven': 45000, 'fuel': 'Petrol', 'transmission': 'Manual', 'owner': 'First Owner', 'seller_type': 'Dealer', 'brand': 'Hyundai'}
   Données transformées pour le modèle : {'year': 2015, 'km_driven': 45000, 'fuel': 1, 'transmission': 1, 'owner': 0, 'seller_type': 0, 'brand': 0}
   Prix prédit : 550000.0
   ```

3. Modifiez les caractéristiques utilisateur directement dans le script pour tester d'autres scénarios.

---

### **Étape 4 : Évaluer le modèle**
1. Lancez le script `model_evaluation.py` pour évaluer les performances du modèle sur un jeu de test :
   ```bash
   python model_evaluation.py
   ```

2. Le script affichera :
   - Un tableau comparant les prix réels et prédits.
   - Une visualisation graphique des différences entre les prédictions et les valeurs réelles.

3. Les résultats peuvent être sauvegardés dans un fichier CSV pour une analyse ultérieure.

---

### **Étape 5 : Lancer l'API**
1. Une fois le modèle correctement enregistré et testé, lancez le serveur FastAPI pour servir le modèle :
   ```bash
   uvicorn main:app --reload
   ```

2. Testez les différents endpoints via l'interface Swagger :
   [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## Tester en Ligne de Commande

### 1. Tester l'Endpoint `/status`

Requête :

```bash
curl -X GET http://127.0.0.1:8000/status
```

Réponse (exemple) :

```json
{
  "status": "Model is ready for prediction"
}
```

### 2. Tester l'Endpoint `/predict`

Requête :

```bash
curl -X POST http://127.0.0.1:8000/predict \
-H "Content-Type: application/json" \
-d '{
    "year": 2015,
    "km_driven": 45000,
    "fuel": "Petrol",
    "transmission": "Manual",
    "owner": "First Owner",
    "seller_type": "Dealer",
    "brand": "Hyundai"
}'
```

Réponse (exemple) :

```json
{
  "predicted_selling_price": 550000.0
}
```

### 3. Tester l'Endpoint `/metadata`

Requête :

```bash
curl -X GET http://127.0.0.1:8000/metadata
```

Réponse (exemple) :

```json
{
  "model_uri": "models:/RandomForestPipeline/1",
  "fuel_mapping": {
    "Diesel": 0,
    "Petrol": 1,
    "LPG": 2,
    "CNG": 3,
    "Electric": 4
  },
  "transmission_mapping": {
    "Automatic": 0,
    "Manual": 1
  },
  "status": "Modèle chargé avec succès"
}
```

### 4. Tester l'Endpoint `/explain`

Requête :

```bash
curl -X POST http://127.0.0.1:8000/explain \
-H "Content-Type: application/json" \
-d '{
    "year": 2015,
    "km_driven": 45000,
    "fuel": "Petrol",
    "transmission": "Manual",
    "owner": "First Owner",
    "seller_type": "Dealer",
    "brand": "Hyundai"
}'
```

Réponse (exemple) :

```json
{
  "base_value": 287795.79,
  "prediction": 495970.11,
  "feature_impact": {
    "year": "Contribution positive importante : +286507.35",
    "km_driven": "Réduction due au kilométrage élevé : -88333.03",
    "fuel": "Pas d'impact significatif",
    "seller_type": "Pas d'impact significatif",
    "transmission": "Pas d'impact significatif",
    "owner": "Pas d'impact significatif",
    "brand": "Pas d'impact significatif"
  }
}
```

### 5. Tester l'Endpoint `/explain-visual`

Requête :

```bash
curl -X POST http://127.0.0.1:8000/explain-visual \
-H "Content-Type: application/json" \
-d '{
    "year": 2015,
    "km_driven": 45000,
    "fuel": "Petrol",
    "transmission": "Manual",
    "owner": "First Owner",
    "seller_type": "Dealer",
    "brand": "Hyundai"
}'
```

Réponse :

Un graphique visuel s'ouvre pour montrer les impacts des caractéristiques sur la prédiction.


---

## Dépannage

1. **Erreur : Modèle non trouvé**

   - Assurez-vous que le serveur MLflow est démarré.
   - Vérifiez l'URI du modèle dans MLflow (par exemple, `models:/RandomForestPipeline/1`).

2. **Erreur : Données invalides**

   - Vérifiez les valeurs fournies dans la requête (par exemple, les valeurs pour `fuel` ou `transmission`).
   - Consultez l'endpoint `/metadata` pour les valeurs acceptées.

3. **Erreur interne du serveur**

   - Lancez FastAPI en mode debug pour afficher plus de détails :
     ```bash
     uvicorn main:app --reload
     ```

---

## Prochaines Étapes

1. Ajouter des logs pour les prédictions dans MLflow.
2. Dockeriser l'application pour faciliter le déploiement.
3. Intégrer des tests automatisés pour valider les endpoints.
