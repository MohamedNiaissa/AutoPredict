# Projet : API FastAPI avec Modèle MLflow

Ce projet implémente une API REST avec **FastAPI** pour servir un modèle machine learning logué dans **MLflow**. L'utilisateur peut envoyer des caractéristiques de véhicules via une requête POST pour obtenir une estimation du prix de vente.

---

## Fonctionnalités

- **Endpoint ****`/status`** : Permet de vérifier si l'API est prête.
- **Endpoint ****`/predict`** : Permet d'envoyer les caractéristiques d'un véhicule et d'obtenir une estimation de son prix de vente.
- **Endpoint ****`/metadata`** : Fournit des informations sur le modèle et les mappages utilisés pour la transformation des données.

---

## Prérequis

1. **Python 3.9+**
2. Installer les dépendances Python :
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
```

---

## **Guide : Enregistrer et Tester un Modèle avec MLflow**

### **Étape 1 : Préparer les données**
Assurez-vous que les données nécessaires au modèle sont disponibles dans le dossier `data/` (exemple : `cartest.csv`).

---

### **Étape 2 : Entraîner et enregistrer le modèle**
1. Naviguez dans le dossier contenant les scripts :
   ```bash
   cd scripts/OneHot
   ```

2. Lancez le script `train_model.py` pour entraîner et enregistrer le modèle dans MLflow :
   ```bash
   python train_model.py
   ```

3. Une fois l'exécution terminée :
   - Un nouveau modèle sera enregistré dans MLflow.
   - Les métriques d'entraînment, comme le **Mean Squared Error**, seront affichées.
   - Vous pouvez consulter le modèle et les artefacts associés dans l'interface MLflow :
     [http://127.0.0.1:8080](http://127.0.0.1:8080)

---

### **Étape 3 : Tester le modèle**
1. Lancez le script `test_model.py` pour vérifier le fonctionnement du modèle avec des données de test :
   ```bash
   python test_model.py
   ```

2. Le script affichera :
   - Les données de test utilisées.
   - Les prédictions générées par le modèle.

---

### **Étape 4 : Lancer l'API**
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

