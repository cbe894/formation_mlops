from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Charger le modèle préentraîné
model = joblib.load('rf_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')  # Assurez-vous que ce return est dans une fonction

# Définir la route principale pour la prédiction
@app.route('/predict', methods=['POST'])
def predict():
    # Utiliser request.form pour extraire les données du formulaire HTML
    age = float(request.form['Age'])
    account_manager = int(request.form['Account_Manager'])
    years = float(request.form['Years'])
    num_sites = int(request.form['Num_Sites'])

    # Créer un tableau numpy pour les données de prédiction
    features = np.array([[age, account_manager, years, num_sites]])

    # Effectuer la prédiction
    prediction = model.predict(features)

    # Convertir la prédiction en un format compréhensible
    result = int(prediction[0] > 0.5)

    return jsonify({'churn_prediction': result})  # Le return doit être ici aussi dans une fonction

# Fonction pour lancer le serveur Flask
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
