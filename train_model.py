# Fichier: train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score
import joblib
import mlflow
import mlflow.sklearn
import warnings
import os

warnings.filterwarnings('ignore')

# --- GESTION DES CHEMINS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data")
MODEL_PATH = os.path.join(BASE_DIR, "model")

# --- NOUVELLE CONFIGURATION MLFLOW (avec support SQLite pour le Registre) ---
# Chemin du fichier de base de données SQLite pour le Registre de Modèles
SQL_BACKEND_URI = "sqlite:///" + os.path.join(BASE_DIR, "mlflow_backend.db")
# Chemin pour les artefacts (les fichiers réels comme les modèles .pkl)
ARTIFACT_URI = "file://" + os.path.join(BASE_DIR, "mlruns")

# 1. Configurer le Tracking URI pour MLflow.
# Ceci permet d'utiliser le Registre de Modèles.
mlflow.set_tracking_uri(uri=SQL_BACKEND_URI)
mlflow.set_experiment("bank-churn-prediction")

print("Chargement des donnees...")
try:
    # Utilise le chemin absolu pour charger le fichier
    df = pd.read_csv(os.path.join(DATA_PATH, "bank_churn.csv"))
except FileNotFoundError:
    print(f"Erreur: Le fichier 'bank_churn.csv' est introuvable dans {DATA_PATH}.")
    exit()

# Créer le répertoire 'model' s'il n'existe pas
os.makedirs(MODEL_PATH, exist_ok=True)

# 1. PRÉ-TRAITEMENT DES DONNÉES
categorical_cols = ['Geography', 'Gender']
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# 2. Séparation features/target
X = df.drop('Exited', axis=1)
y = df['Exited']

# Split train/test (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"\nTrain : {len(X_train)} lignes")

# 3. Entrainement avec MLflow tracking
with mlflow.start_run(run_name="random-forest-retrain") as run:
    
    # Parametres du modele
    params = {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 5,
        'random_state': 42
    }
    
    # Entrainement
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Calcul des metriques
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    
    # Log des parametres et metriques dans MLflow
    mlflow.log_params(params)
    mlflow.log_metrics({"f1_score": f1, "roc_auc": auc})
    
    # 4. Enregistrement du modèle
    
    # Enregistrement dans MLflow Model Registry (cela nécessite le backend SQLite)
    mlflow.sklearn.log_model(
        model,
        "model_artifact", 
        registered_model_name="bank-churn-classifier"
    )
    
    # Sauvegarde locale du modele (pour l'API FastAPI, qui utilise joblib.load)
    joblib.dump(model, os.path.join(MODEL_PATH, "churn_model.pkl"))
    
    print("\n" + "="*50)
    print("RESULTATS DE L'ENTRAINEMENT")
    print("="*50)
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC AUC  : {auc:.4f}")
    print("="*50)
    print(f"Modèle enregistré sous 'model/churn_model.pkl'")
    print(f"MLflow Run ID: {run.info.run_id}")