import requests, random

url = 'http://127.0.0.1:5000/predict'

def generate_random_json():
    return {
        'Age': random.randint(1, 100),  # Génère un entier aléatoire entre 1 et 100
        'Account_Manager': random.randint(0, 50),   # Génère un nombre flottant aléatoire entre 0 et 50
        'Years': random.randint(10, 500), # Génère un entier aléatoire entre 10 et 500
        'Num_Sites': random.randint(100, 1000) # Génère un flottant aléatoire entre 100 et 1000
    }

# Boucle pour générer et afficher 20 objets JSON
for i in range(20):
    data = generate_random_json()
    print(data)
    response = requests.post(url, json=data)
    print(response.json())

# Afficher la réponse du serveur
print(response.json())