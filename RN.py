# RN.py

import re
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# -Ici on vas nettoyer les tweets
def clean_tweet(tweet):
    tweet = tweet.lower()
    tweet = re.sub(r"http\S+", "", tweet)
    tweet = re.sub(r"@\w+", "", tweet)
    tweet = re.sub(r"#\w+", "", tweet)
    tweet = re.sub(r"rt\s?:", "", tweet)
    tweet = re.sub(r"[^\w\s]", "", tweet)
    tweet = re.sub(r"\s+", " ", tweet).strip()
    return tweet

# --Ici on vas charger les données
tweets_data = []
with open("balanced_tweets_200k.csv", "r", encoding="latin1") as f:
    reader = csv.reader(f)
    for row in reader:
        if len(row) >= 6:
            raw_label = row[0]
            tweet = row[5]

            if raw_label == "0":
                label = 0
            elif raw_label == "4": # juste pour le cas de sentiment positif et on norme à 1
                label = 1
            else:
                continue

            tweets_data.append((clean_tweet(tweet), label))

# Ici on vas extraire les colonnes
texts = [t[0] for t in tweets_data if isinstance(t[0], str) and len(t[0].strip()) > 0]
labels = [t[1] for t in tweets_data if isinstance(t[0], str) and len(t[0].strip()) > 0]

# La on vas vectoriser les données
vectorizer = TfidfVectorizer(max_features=512, stop_words=None) #la on vas utiliser 512 features
X = vectorizer.fit_transform(texts).toarray() # # la on vas transformer les textes en vecteurs
y = torch.tensor(labels).float()# # on vas transformer les labels en tenseurs

# --- Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42) # la o sépare les données en train et test
X_train = torch.tensor(X_train).float()#  on vas transformer les données d'entraînement en tenseurs
X_test = torch.tensor(X_test).float()#idem test
y_train = y_train.view(-1, 1)#  vas transformer les labels d'entraînement en tenseurs
y_test = y_test.view(-1, 1)#idem test

# la on vas créer le modèle
class SentimentNN(nn.Module):
    def __init__(self):
        super(SentimentNN, self).__init__()
        self.fc1 = nn.Linear(512, 128)# la on vas créer la première couche linéaire
        self.relu1 = nn.ReLU()# la on vas ajouter une activation ReLU
        self.drop1 = nn.Dropout(0.3)# la on vas ajouter une couche de dropout pour éviter le sur-apprentissage
        
        self.fc2 = nn.Linear(128,64)#idem pour la deuxième couche linéaire
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(0.3)
        
        self.fc3 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid() # sigmoid pour la sortie binaire

    def forward(self, x):# La on apllique ces fonctions dans l'ordre
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.drop1(x)
        
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.drop2(x)
        
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x

model = SentimentNN()# la on vas créer une instance du modèle
criterion = nn.BCELoss()# la on vas utiliser la perte binaire pour la classification binaire
optimizer = optim.Adam(model.parameters(), lr=0.01)# # la on vas à l'optimiseur Adam avec un apprentissage de 0.001 (c'est lent)

# la on s'entraîne

print("⚡ Entraînement du modèle...")
for epoch in range(30):# on vas faire 5 époques d'entraînement ( peu pour commencer)
    model.train()# il s'entraîne
    optimizer.zero_grad()# on réinitialise les gradients
    outputs = model(X_train)
    loss = criterion(outputs, y_train)# la on calcule la perte
    loss.backward()
    optimizer.step()# on met à jour les poids du modèle
    print(f"Epoch {epoch+1}/5 - Loss: {loss.item():.4f}")

# 
model.eval()# il s'évalue
y_pred = model(X_test).detach().numpy()# la on vas faire des prédictions sur les données de test
y_pred_label = [1 if p >= 0.5 else 0 for p in y_pred]# on vas transformer les prédictions en labels binaires

print("\n Résultats sur le test :")
print(classification_report(y_test, y_pred_label))

# la ça vas afficher la matrice de confusion
cm = confusion_matrix(y_test, y_pred_label)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["négatif", "positif"])
disp.plot(cmap=plt.cm.Blues)
plt.title("Matrice de confusion")
plt.show()

# ça c'est pour tester par nous même sans entraînement
def predict_sentiment(tweet):
    cleaned = clean_tweet(tweet)
    vec = vectorizer.transform([cleaned]).toarray()
    with torch.no_grad():
        output = model(torch.tensor(vec).float())
    return "positif" if output.item() >= 0.5 else "négatif"

if __name__ == "__main__":
    print("Exemple :", predict_sentiment("i eat this cheat"))

#dans RNplus et RNpluplus je commente que ce qui change