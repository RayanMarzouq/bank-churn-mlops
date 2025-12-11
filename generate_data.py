# Fichier: generate_data.py
import pandas as pd
import numpy as np
import os # NOUVEL IMPORT
from sklearn.model_selection import train_test_split

np.random.seed(42)
n_samples = 12000 
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA_PATH, exist_ok=True) # Assure que le dossier 'data' existe

# Génération des features (caractéristiques)
data = {
    'CreditScore': np.random.randint(300, 850, n_samples),
    'Age': np.random.randint(18, 80, n_samples),
    'Tenure': np.random.randint(0, 11, n_samples),
    'Balance': np.random.uniform(0, 200000, n_samples),
    'NumOfProducts': np.random.randint(1, 5, n_samples),
    'HasCrCard': np.random.choice([0, 1], n_samples),
    'IsActiveMember': np.random.choice([0, 1], n_samples),
    'EstimatedSalary': np.random.uniform(20000, 150000, n_samples),
    # Variables catégorielles (pour OHE dans train_model.py)
    'Geography': np.random.choice(['France', 'Spain', 'Germany'], n_samples, p=[0.5, 0.25, 0.25]),
    'Gender': np.random.choice(['Female', 'Male'], n_samples),
}

# Target (Exited/Churn)
churn_prob = (
    (1 - (data['Gender'] == 'Male').astype(int) * 0.1) * 0.1 +
    (1 - data['IsActiveMember']) * 0.3 +
    (data['NumOfProducts'] == 1) * 0.2 +
    (data['Age'] > 60) * 0.15 
)
churn_prob = np.clip(churn_prob, 0.05, 0.9) 

data['Exited'] = (np.random.random(n_samples) < churn_prob).astype(int)
df_full = pd.DataFrame(data)

# --- DIVISION ET SAUVEGARDE DES TROIS FICHIERS REQUIS ---

# 1. Dataset pour l'Entraînement initial
df_train = df_full.sample(n=10000, random_state=42)
df_train.to_csv(os.path.join(DATA_PATH, 'bank_churn.csv'), index=False)

# 2. Dataset de Référence (base du monitoring)
df_ref = df_full.sample(n=2000, random_state=43)
df_ref.to_csv(os.path.join(DATA_PATH, 'bank_churn_reference.csv'), index=False)

# 3. Dataset de Production (avec dérive introduite artificiellement)
df_prod = df_full.sample(n=2000, random_state=44).copy()
df_prod['Age'] = df_prod['Age'] + 5 
df_prod['Age'] = np.clip(df_prod['Age'], 18, 80)

df_prod.to_csv(os.path.join(DATA_PATH, 'bank_churn_production.csv'), index=False)

print(f"Dataset total créé : {len(df_full)} lignes")
print(f"Fichiers de données MLOps créés dans {DATA_PATH}.")