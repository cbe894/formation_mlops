# Importation des bibliothèques nécessaires
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def train_random_forest(data, target_column, test_size=0.2, random_state=42, n_estimators=100):
    """
    Entraîne un modèle de Random Forest et évalue ses performances.

    Parameters:
    - data: DataFrame contenant les données.
    - target_column: Nom de la colonne cible.
    - test_size: Proportion des données à utiliser pour le test.
    - random_state: Pour la reproductibilité des résultats.
    - n_estimators: Nombre d'arbres dans la forêt.

    Returns:
    - model: Le modèle Random Forest entraîné.
    """

    # enlever les colonnes
    data = data.drop(['Onboard_date', 'Location', 'Company'], axis=1)

    print(list(data.columns))
    print(list(data.dtypes))

    # Supposons que la dernière colonne soit la variable cible 'target'
    # Assurez-vous d'ajuster cette ligne pour correspondre à votre dataset
    X = data[['Age', 'Account_Manager', 'Years', 'Num_Sites']]  # Toutes les colonnes sauf la dernière (features)
    y = data[target_column]  # Dernière colonne (target)

    # Diviser le dataset en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Initialiser le modèle Random Forest
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)

    # Entraîner le modèle sur les données d'entraînement
    model.fit(X_train, y_train)

    return model


def pred_random_forest(model, X_test, y_test):
    """
    Teste un modèle Random Forest sur un ensemble de test et affiche les performances.

    Parameters:
    - model: Modèle Random Forest déjà entraîné.
    - X_test: Caractéristiques de l'ensemble de test.
    - y_test: Cible de l'ensemble de test.

    Returns:
    - accuracy: La précision du modèle sur l'ensemble de test.
    """

    # Faire des prédictions sur l'ensemble de test
    y_pred = model.predict(X_test)

    # Évaluer les performances du modèle
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy of the model: {accuracy:.2f}")

    # Afficher le rapport de classification
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Afficher la matrice de confusion
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return accuracy

if __name__ == "__main__":
    # Charger les données à partir d'un fichier CSV
    # Remplace 'votre_fichier.csv' par le chemin vers ton fichier CSV
    data = pd.read_csv('data/customer_churn.csv')

    # Entraîner le modèle Random Forest
    model_trained = train_random_forest(data, target_column='Churn')
    print("Le modele entrainé :", model_trained)

    # Assurez-vous d'ajuster cette ligne pour correspondre à votre dataset
    X = data[['Age', 'Account_Manager', 'Years', 'Num_Sites']]  # Toutes les colonnes sauf la dernière (features)
    y = data['Churn']  # Dernière colonne (target)

    # Diviser les données en ensemble d'entraînement et de test (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # faire la prediction
    pred = pred_random_forest(model = model_trained, X_test = X_test, y_test = y_test)
    print("La prévision :", pred)

