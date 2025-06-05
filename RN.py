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

# --- Nettoyage ---
def clean_tweet(tweet):
    tweet = tweet.lower()
    tweet = re.sub(r"http\S+", "", tweet)
    tweet = re.sub(r"@\w+", "", tweet)
    tweet = re.sub(r"#\w+", "", tweet)
    tweet = re.sub(r"rt\s?:", "", tweet)
    tweet = re.sub(r"[^\w\s]", "", tweet)
    tweet = re.sub(r"\s+", " ", tweet).strip()
    return tweet

# --- Extraction depuis CSV ---
tweets_data = []
with open("MMM.csv", "r", encoding="latin1") as f:
    reader = csv.reader(f)
    for row in reader:
        if len(row) >= 6:
            raw_label = row[0]
            tweet = row[5]

            if raw_label == "0":
                label = 0
            elif raw_label == "4":
                label = 1
            else:
                continue

            tweets_data.append((clean_tweet(tweet), label))

# --- Extraction des colonnes ---
texts = [t[0] for t in tweets_data if isinstance(t[0], str) and len(t[0].strip()) > 0]
labels = [t[1] for t in tweets_data if isinstance(t[0], str) and len(t[0].strip()) > 0]

# --- Vectorisation ---
vectorizer = TfidfVectorizer(max_features=1024, stop_words=None)
X = vectorizer.fit_transform(texts).toarray()
y = torch.tensor(labels).float()

# --- Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
X_train = torch.tensor(X_train).float()
X_test = torch.tensor(X_test).float()
y_train = y_train.view(-1, 1)
y_test = y_test.view(-1, 1)

# --- Modèle ---
class SentimentNN(nn.Module):
    def __init__(self):
        super(SentimentNN, self).__init__()
        self.fc1 = nn.Linear(1024, 1024)
        self.relu1 = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(1024, 128)
        self.relu2 = nn.ReLU()
        
        self.fc3 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu2(x)
        
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x

model = SentimentNN()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# --- Entraînement ---
print("⚡ Entraînement du modèle...")
for epoch in range(5):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}/5 - Loss: {loss.item():.4f}")

# --- Évaluation ---
model.eval()
y_pred = model(X_test).detach().numpy()
y_pred_label = [1 if p >= 0.5 else 0 for p in y_pred]

print("\n Résultats sur le test :")
print(classification_report(y_test, y_pred_label))

# --- Matrice de confusion ---
cm = confusion_matrix(y_test, y_pred_label)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["négatif", "positif"])
disp.plot(cmap=plt.cm.Blues)
plt.title("Matrice de confusion")
plt.show()

# --- Prédiction personnalisée ---
def predict_sentiment(tweet):
    cleaned = clean_tweet(tweet)
    vec = vectorizer.transform([cleaned]).toarray()
    with torch.no_grad():
        output = model(torch.tensor(vec).float())
    return "positif" if output.item() >= 0.5 else "négatif"

if __name__ == "__main__":
    print("Exemple :", predict_sentiment("i eat this cheat"))
