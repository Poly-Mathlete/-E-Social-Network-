# algomattis.py -- Version 1 du programme de regression linéaire -- Ne donne pas des résultats probants

import re
from sklearn.model_selection import train_test_split # Pour découper les donner en train/test
from sklearn.feature_extraction.text import TfidfVectorizer # Pour transformer les textes en vecteurs 
from sklearn.linear_model import LinearRegression # Pour créer un modèle de régression linéaire

from tweetsss import tweets_data # On importe les tweets 

# Extraction
raw_tweets = [t[0] for t in tweets_data] # Les tweets de base
score_pos = [t[1] for t in tweets_data] # Score positif
score_neg = [t[2] for t in tweets_data] # Score négatif 
labels = [t[3] for t in tweets_data] # Classification globale 

# Nettoyage, on enlève les liens, les mentions, les #, et les autres résidus"
def clean_tweet(tweet):
    tweet = tweet.lower()
    tweet = re.sub(r"http\S+", "", tweet)
    tweet = re.sub(r"@\w+", "", tweet)
    tweet = re.sub(r"#\w+", "", tweet)
    tweet = re.sub(r"rt\s?:", "", tweet)
    tweet = re.sub(r"[^\w\s]", "", tweet)
    tweet = re.sub(r"\s+", " ", tweet).strip()
    return tweet

cleaned_tweets = [clean_tweet(t) for t in raw_tweets] 

# Vectorisation
vectorizer = TfidfVectorizer(max_features=1000) # On transforme en vecteur numérique avec maximum 1000 mots 
X = vectorizer.fit_transform(cleaned_tweets) # Matrice des vecteurs pour chaque tweet 
ypos = score_pos # Cible poistive 
yneg = score_neg # Cible négative 

# Split & modèle : sur positif et négatif 
X_train, X_test, y_trainpos, y_testpos = train_test_split(X, ypos, test_size=0.20, random_state=42) # On découpe en 80% train et 20% test
X_train, X_test, y_trainneg, y_testneg = train_test_split(X, yneg, test_size=0.20, random_state=42)
modelpos = LinearRegression() # On fait la régression linéaire 
modelneg = LinearRegression()
modelpos.fit(X_train, y_trainpos) # On entraine le modèle
modelneg.fit(X_train, y_trainneg)

# Évaluation - On prédit les scores positifs et négatifs des tweets de test avec les modèles entraînés.
y_predpos = modelpos.predict(X_test)
y_predneg = modelneg.predict(X_test)

def predict_sentiment(tweet):
    cleaned = clean_tweet(tweet)
    vec = vectorizer.transform([cleaned])
    posscore=modelpos.predict(vec)[0]
    negscore=modelneg.predict(vec)[0]
    if posscore > 0.05+negscore:
        return ("Positif",posscore,negscore)
    elif negscore > 0.05+posscore:
        return ("Négatif",posscore,negscore)
    else:
        return ("Neutre",posscore,negscore)
    
'''
La fonction ci-dessus prend un tweet en texte brut, le nettoie, le vectorise, puis prédit les scores positifs et négatifs.
Ensuite :
	•	si le score positif dépasse le score négatif de plus de 0.05, on dit que c’est positif
	•	si le score négatif dépasse le score positif de plus de 0.05, c’est négatif
	•	sinon, on considère le tweet neutre
Elle retourne un tuple avec le label et les scores.

'''

if __name__ == "__main__":
    print("Entraînement du modèle...")
    from sklearn.metrics import mean_squared_error, r2_score

    print("Score Positif")
    print("MSE :", mean_squared_error(y_testpos, y_predpos))
    print("R² :", r2_score(y_testpos, y_predpos))

    print("Score Négatif")
    print("MSE :", mean_squared_error(y_testneg, y_predneg))
    print("R² :", r2_score(y_testneg, y_predneg))

    print("Exemple de prédiction :")
    print(predict_sentiment("J'en ai marre de ce bordel !"))

'''
Donne les résultats propre au modèle 


Remarque : ce programme renvoie une erreur à cause du fait qu'il faut avoir le fichier tweets_data en local 
On ne peut pasmettre le fichier sur le got car il pèse plus de 100 MO 
'''