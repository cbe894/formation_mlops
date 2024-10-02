from flask import Flask, request, jsonify, render_template
from prometheus_client import Counter, Histogram, generate_latest
import joblib
import numpy as np

app = Flask(__name__)

# Créer un compteur pour suivre le nombre de requêtes
REQUEST_COUNT = Counter('flask_app_requests_total',
                        'Total number of requests to the Flask app',
                        ['method', 'endpoint'])

# Créer un histogramme pour mesurer la latence des requêtes
REQUEST_LATENCY = Histogram('flask_app_request_latency_seconds',
                            'Request latency in seconds',
                            ['endpoint'])


# Charger le modèle préentraîné
model = joblib.load('rf_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')  # Assurez-vous que ce return est dans une fonction

# Définir la route principale pour la prédiction
@app.route('/predict', methods=['POST'])
def predict():
    REQUEST_LATENCY.labels(endpoint='/predict').time()  # Mesurer la latence
    REQUEST_COUNT.labels(method='POST', endpoint='/predict').inc()  # Incrémenter le compteur de requêtes

    data = request.json
    age = data['Age']
    account_manager = data['Account_Manager']
    years = data['Years']
    num_sites = data['Num_Sites']


    # Créer un tableau numpy pour les données de prédiction
    features = np.array([[age, account_manager, years, num_sites]])

    # Effectuer la prédiction
    prediction = model.predict(features)

    # Convertir la prédiction en un format compréhensible
    result = int(prediction[0] > 0.5)

    return jsonify({'churn_prediction': result})  # Le return doit être ici aussi dans une fonction


# Point d'exposition des métriques Prometheus
@app.route('/metrics')
def metrics():
    return generate_latest(), 200, {'Content-Type': 'text/plain; charset=utf-8'}


# Fonction pour lancer le serveur Flask
if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
