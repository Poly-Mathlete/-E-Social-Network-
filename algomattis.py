# algomattis.py

import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report

from tweetsss import tweets_data

# Extraction
raw_tweets = [t[0] for t in tweets_data]
score_pos = [t[1] for t in tweets_data]
score_neg = [t[2] for t in tweets_data]
labels = [t[3] for t in tweets_data]

# Nettoyage
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
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(cleaned_tweets)
ypos = score_pos
yneg = score_neg

# Split & modèle
X_train, X_test, y_trainpos, y_testpos = train_test_split(X, ypos, test_size=0.20, random_state=42)
X_train, X_test, y_trainneg, y_testneg = train_test_split(X, yneg, test_size=0.20, random_state=42)
modelpos = LinearRegression()
modelneg = LinearRegression()
modelpos.fit(X_train, y_trainpos)
modelneg.fit(X_train, y_trainneg)

# Évaluation
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
    

if __name__ == "__main__":
    print("⚙ Entraînement du modèle...")
    from sklearn.metrics import mean_squared_error, r2_score

    print("📈 Score Positif")
    print("MSE :", mean_squared_error(y_testpos, y_predpos))
    print("R² :", r2_score(y_testpos, y_predpos))

    print("📉 Score Négatif")
    print("MSE :", mean_squared_error(y_testneg, y_predneg))
    print("R² :", r2_score(y_testneg, y_predneg))

    print("Exemple de prédiction :")
    print(predict_sentiment("J'en ai marre de ce bordel !"))
