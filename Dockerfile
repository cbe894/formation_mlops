# Utiliser une image officielle de Python 3.9
FROM python:3.11-slim

# Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Copier le fichier requirements.txt dans le conteneur
COPY requirements.txt requirements.txt

# Installer les dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Copier tout le contenu du projet dans le conteneur
COPY . .

# Exposer le port sur lequel l'application va s'exécuter
EXPOSE 5000

# Commande pour exécuter l'application Flask
CMD ["python", "app.py"]