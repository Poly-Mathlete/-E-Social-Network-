

import re
import csv
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
# Extraction et nettoyage des tweets
tweets_data = []
with open("balanced_tweets_200k.csv", "r", encoding="latin1") as f:
    reader = csv.reader(f)
    for row in reader:
        if len(row) >= 6:
            raw_label = row[0]
            tweet = row[5]

            if raw_label == "0":
                label = "négatif"
            elif raw_label == "4":
                label = "positif"
            else:
                continue  # On ignore les autres labels

            tweets_data.append((tweet, label))

# colonnes
raw_tweets = [t[0] for t in tweets_data]
labels = [t[1] for t in tweets_data]

# on nettoie les tweets
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

# on les vectorisation TF-IDF
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(cleaned_tweets)
y = labels

# on entraîne le modèle
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)

# prediction
def predict_sentiment(tweet):
    cleaned = clean_tweet(tweet)
    vec = vectorizer.transform([cleaned])
    return model.predict(vec)[0]

# --- Évaluation ---
if __name__ == "__main__":
    print("⚙ Entraînement du modèle...")
    print(classification_report(y_test, model.predict(X_test)))
    

# Matrice de confusion
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred, labels=["négatif", "positif"])        
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["négatif", "positif"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Matrice de confusion")
    plt.show()

    print("Exemple de prédiction :")
    print(predict_sentiment("shut the fuck up"))  
