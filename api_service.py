# Fichier: api_service.py (Version Finale Corrigée pour la Synchronisation)
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import warnings
import os

warnings.filterwarnings('ignore')

# --- GESTION DES CHEMINS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE_PATH = os.path.join(BASE_DIR, "model", "churn_model.pkl")

# --- LISTE DES COLONNES ATTENDUES PAR LE MODELE (CRUCIAL) ---
# Cette liste doit correspondre exactement aux colonnes d'entraînement après OHE et suppression de 'Exited'.
# L'ordre doit être conservé.
EXPECTED_FEATURES = [
    'CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 
    'HasCrCard', 'IsActiveMember', 'EstimatedSalary',
    'Geography_Germany', 
    'Geography_Spain',   
    'Gender_Male'        
]

# --- 1. CHARGEMENT DU MODÈLE ET INITIALISATION ---
model = None
try:
    model = joblib.load(MODEL_FILE_PATH)
    print(f"Modèle 'churn_model.pkl' chargé avec succès depuis {MODEL_FILE_PATH}.")
    
    # DIAGNOSTIC : Affiche les features vues par le modèle si possible
    if hasattr(model, 'feature_names_in_'):
        print(f"Features vues par le modèle entraîné: {list(model.feature_names_in_)}")
    else:
        print("Avertissement: Impossible d'extraire les noms de features du modèle.")

except FileNotFoundError:
    print(f"Erreur: Le fichier modèle est introuvable à {MODEL_FILE_PATH}. Exécutez 'train_model.py'.")
except Exception as e:
    print(f"Erreur lors du chargement du modèle: {e}")

# Initialisation de l'application FastAPI
app = FastAPI(title="Bank Churn Prediction API", version="1.0.0")

# --- 2. DÉFINITION DE LA STRUCTURE DES DONNÉES D'ENTRÉE (Pydantic) ---
class ChurnFeatures(BaseModel):
    CreditScore: int
    Geography: str
    Gender: str
    Age: int
    Tenure: int
    Balance: float
    NumOfProducts: int
    HasCrCard: int
    IsActiveMember: int
    EstimatedSalary: float

# --- 3. DÉFINITION DU POINT DE TERMINAISON DE PRÉDICTION ---
@app.post("/predict")
def predict_churn(data: ChurnFeatures):
    if model is None:
        return {"error": "Modèle non chargé, veuillez vérifier le chemin et l'entraînement."}

    # 1. Préparation du DataFrame
    input_data = data.dict()
    df_input = pd.DataFrame([input_data])
    
    # 2. Application du ONE-HOT ENCODING (OHE)
    categorical_cols = ['Geography', 'Gender']
    df_encoded = pd.get_dummies(df_input, columns=categorical_cols, drop_first=True)
    
    # 3. SYNCHRONISATION DES FEATURES (Ajout des colonnes manquantes à 0 et ordonnancement)
    
    # Ajouter les colonnes manquantes (celles qui n'étaient pas dans l'input mais sont attendues, ex: Geography_Spain)
    for feature in EXPECTED_FEATURES:
        if feature not in df_encoded.columns:
            df_encoded[feature] = 0
            
    # Filtrer et ordonner les colonnes pour correspondre EXACTEMENT au modèle
    df_final = df_encoded.filter(items=EXPECTED_FEATURES)

    # VÉRIFICATION FINALE (diagnostic)
    if df_final.shape[1] != len(EXPECTED_FEATURES):
        return {"error": "Erreur critique de dimension. Le DataFrame final n'a pas le bon nombre de colonnes."}

    try:
        # PRÉDICTION
        prediction_proba = model.predict_proba(df_final)[0, 1]
        prediction_class = int(model.predict(df_final)[0])
    except Exception as e:
        # Capture les erreurs spécifiques à la prédiction (souvent liées à la structure des données)
        return {"error": f"Erreur lors de l'appel predict: {str(e)}", 
                "df_columns": list(df_final.columns),
                "df_shape": str(df_final.shape)}


    # Retourner le résultat
    return {
        "churn_prediction": prediction_class,
        "churn_probability": round(prediction_proba, 4),
        "input_data_received": input_data
    }

# --- 4. POINT DE TERMINAISON DE TEST ---
@app.get("/")
def home():
    return {"health_check": "API est opérationnelle."}