# Fichier: detect_drift.py
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
import sys
import os # NOUVEL IMPORT

# --- GESTION DES CHEMINS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data")

# --- CONFIGURATION ---
ALERT_THRESHOLD = 0.05

# Charger les données de référence et de production
try:
    df_ref = pd.read_csv(os.path.join(DATA_PATH, "bank_churn_reference.csv"))
    df_prod = pd.read_csv(os.path.join(DATA_PATH, "bank_churn_production.csv"))
except FileNotFoundError:
    print(f"Erreur: Fichiers de données manquants. Exécutez 'python generate_data.py'.")
    sys.exit(1) # Signalons une erreur grave qui justifie le ré-entraînement ou une intervention

# 1. Création et exécution du rapport
data_drift_report = Report(metrics=[DataDriftPreset()])
data_drift_report.run(reference_data=df_ref, current_data=df_prod, column_mapping=None)

# 2. Extraction du statut
report_json = data_drift_report.as_dict()
dataset_drift_status = report_json['metrics'][0]['result']['dataset_drift']
drifted_columns = report_json['metrics'][0]['result']['number_of_drifted_columns']
total_columns = report_json['metrics'][0]['result']['number_of_columns']
drifted_columns_ratio = drifted_columns / total_columns

print(f"\n--- RAPPORT DE DÉRIVE (Evidently) ---")
print(f"Statut global de dérive : {dataset_drift_status}")
print(f"Colonnes en dérive : {drifted_columns} / {total_columns}")

# --- LOGIQUE D'ALERTE POUR L'ORCHESTRATION ---
if drifted_columns_ratio >= ALERT_THRESHOLD:
    print("\n🚨 ALERTE DE DÉRIVE MAJEURE DÉTECTÉE 🚨")
    print("Code de sortie: 1 (Déclenchement du Ré-entraînement)")
    sys.exit(1) 
else:
    print("\n✅ PAS DE DÉRIVE MAJEURE DÉTECTÉE.")
    print("Code de sortie: 0 (Le modèle reste en place)")
    sys.exit(0)