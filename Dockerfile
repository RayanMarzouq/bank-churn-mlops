
# Dockerfile
# Utilise une image Python officielle
FROM python:3.9-slim

# Definir le repertoire de travail
WORKDIR /app

# Copier les fichiers de dependances
COPY requirements.txt .

# Installer les dependances
# --no-cache-dir pour reduire la taille de l'image
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code de l'application
# Copie le dossier app/ et son contenu (main.py, models.py, etc.)
COPY app/ ./app/
# Copie le dossier model/ et le fichier churn_model.pkl
COPY model/ ./model/

# Exposer le port (pour la documentation)
EXPOSE 8000

# Commande pour demarrer l'application
# Le port par defaut est 8000 (ce qui correspond à EXPOSE ci-dessus)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]