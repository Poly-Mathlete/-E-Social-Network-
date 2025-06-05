# reg_lin_improved.py -- Version 2 du programme de régression logistique -- Meilleure version

import re                          # Module pour les expressions régulières - regex (nettoyage de texte)
import csv                         # Module pour lire les fichiers CSV
from sklearn.model_selection import train_test_split      # Pour séparer les données en jeu d'entraînement/test
from sklearn.feature_extraction.text import TfidfVectorizer  # Pour transformer les textes en vecteurs numériques
from sklearn.linear_model import LogisticRegression        # Modèle de classification : régression logistique
from sklearn.metrics import classification_report          # Pour évaluer le modèle avec un rapport de classification

# --- Extraction depuis un fichier CSV ---
tweets_data = []                                        # Liste qui contiendra les tweets et leurs étiquettes (label)
with open("16.csv", "r", encoding="latin1") as f:       # On ouvre le fichier CSV 
    reader = csv.reader(f)                              # On crée un lecteur CSV ligne par ligne
    for row in reader:                                  # Pour chaque ligne brut du fichier :
        if len(row) >= 6:                               # On vérifie que la ligne contient au moins 6 colonnes
            raw_label = row[0]                          # La 1ère colonne est le score déja mis : 0 (négatif) ou 4 (positif)
            tweet = row[5]                              # La 6e colonne contient le texte du tweet

            if raw_label == "0":                        # Si la polarité est 0, on le note comme négatif
                label = "négatif"
            elif raw_label == "4":                      # Si c'est 4, on le note comme positif
                label = "positif"
            else:
                continue                                # Si ce n’est ni 0 ni 4, on ignore la ligne

            tweets_data.append((tweet, label))          # On ajoute le tweet et son label à la liste

# --- Séparation des colonnes tweet / label ---
raw_tweets = [t[0] for t in tweets_data]                # Liste contenant uniquement les textes des tweets
labels = [t[1] for t in tweets_data]                    # Liste contenant uniquement les étiquettes (positif/négatif)

# --- Fonction de nettoyage des tweets ---
def clean_tweet(tweet):
    tweet = tweet.lower()                               # On passe tout en minuscules
    tweet = re.sub(r"http\S+", "", tweet)               # On enlève les URLs
    tweet = re.sub(r"@\w+", "", tweet)                  # On enlève les mentions @
    tweet = re.sub(r"#\w+", "", tweet)                  # On enlève les hashtags
    tweet = re.sub(r"rt\s?:", "", tweet)                # On enlève les "RT :" des retweets car on en tient pas compte ici
    tweet = re.sub(r"[^\w\s]", "", tweet)               # On enlève la ponctuation
    tweet = re.sub(r"\s+", " ", tweet).strip()          # On enlève les espaces en trop
    return tweet

cleaned_tweets = [clean_tweet(t) for t in raw_tweets]   # On nettoie tous les tweets

# --- Vectorisation avec TF-IDF ---
vectorizer = TfidfVectorizer(max_features=1000)         # On transforme les textes en vecteurs (1000 mots max)
X = vectorizer.fit_transform(cleaned_tweets)            # On applique la vectorisation
y = labels                                              # Les labels ne changent pas

# --- Entraînement du modèle de régression logistique ---
X_train, X_test, y_train, y_test = train_test_split(    # On sépare les données : 80% entraînement, 20% test
    X, y, test_size=0.2, random_state=42
)
model = LogisticRegression(max_iter=10000)              # On initialise la régression logistique avec un grand nombre d'itérations
model.fit(X_train, y_train)                             # On entraîne le modèle sur les données d'entraînement

# --- Fonction de prédiction d’un nouveau tweet ---
def predict_sentiment(tweet):
    cleaned = clean_tweet(tweet)                        # On nettoie le tweet
    vec = vectorizer.transform([cleaned])               # On vectorise le tweet nettoyé
    return model.predict(vec)[0]                        # On prédit la polarité (positif/négatif)

# --- Évaluation du modèle (si le fichier est exécuté directement) ---
if __name__ == "__main__":
    print("Entraînement du modèle...")
    print(classification_report(y_test, model.predict(X_test)))  # On affiche les métriques d’évaluation
    print("Exemple de prédiction :")
    print(predict_sentiment("J'en ai marre de ce bordel !"))     # On teste le modèle sur un exemple



'''
Le programme fonctionne bien mais il faut le fichier 16.csv en local !
'''