# Importation des bibliothèques nécessaires
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Charger les données à partir d'un fichier CSV
# Remplace 'votre_fichier.csv' par le chemin vers ton fichier CSV
data = pd.read_csv('data/customer_churn.csv')

# info sur le data frame
print(data.info)
print(list(data.columns))

# enlever les colonnes
data = data.drop(['Onboard_date', 'Location', 'Company'], axis=1)

print(list(data.columns))
print(list(data.dtypes))

# Supposons que la dernière colonne soit la variable cible 'target'
# Assurez-vous d'ajuster cette ligne pour correspondre à votre dataset
X = data[['Age', 'Account_Manager', 'Years', 'Num_Sites']]  # Toutes les colonnes sauf la dernière (features)
y = data['Churn']   # Dernière colonne (target)

print(X.dtypes)

# Diviser les données en ensemble d'entraînement et de test (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialiser le modèle Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Entraîner le modèle sur les données d'entraînement
rf_model.fit(X_train, y_train)

# Faire des prédictions sur l'ensemble de test
y_pred = rf_model.predict(X_test)

# Évaluer les performances du modèle
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of the model: {accuracy:.2f}")

# Afficher le rapport de classification
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Sauvegarder le modèle entraîné dans un fichier .pkl
joblib.dump(rf_model, 'rf_model.pkl')

print("Modèle sauvegardé sous 'rf.pkl'")

# cbeGroupgithub12