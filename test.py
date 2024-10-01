import pandas as pd
from sklearn.model_selection import train_test_split

from training import train_random_forest, pred_random_forest


# test training fonction
def test_train():
    # Charger les données à partir d'un fichier CSV
    # Remplace 'votre_fichier.csv' par le chemin vers ton fichier CSV
    data = pd.read_csv('data/customer_churn.csv')

    # Entraîner le modèle Random Forest
    model_trained = train_random_forest(data, target_column='Churn')

    expected = 'gini'
    assert model_trained.criterion == expected, f"Expected {expected} is not ok"

# test pred fonction
def test_pred():
    # Charger les données à partir d'un fichier CSV
    # Remplace 'votre_fichier.csv' par le chemin vers ton fichier CSV
    data = pd.read_csv('data/customer_churn.csv')

    # Assurez-vous d'ajuster cette ligne pour correspondre à votre dataset
    X = data[['Age', 'Account_Manager', 'Years', 'Num_Sites']]  # Toutes les colonnes sauf la dernière (features)
    y = data['Churn']

    # Diviser les données en ensemble d'entraînement et de test (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Entraîner le modèle Random Forest
    model_trained = train_random_forest(data, target_column='Churn')

    # faire la prediction
    pred = pred_random_forest(model=model_trained, X_test=X_test, y_test=y_test)

    expected = 0.8
    assert pred >= expected, f"Expected {expected}, is not ok"

