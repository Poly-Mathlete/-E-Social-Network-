# algomattis_nn.py

import re
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report

# --- Extraction depuis CSV ---
tweets_data = []
with open("lol.csv", "r", encoding="latin1") as f:
    reader = csv.reader(f)
    for row in reader:
        if len(row) >= 6:
            raw_label = row[0]
            tweet = row[5]

            if raw_label == "0":
                label = 0  # négatif
            elif raw_label == "4":
                label = 1  # positif
            else:
                continue

            tweets_data.append((tweet, label))

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

texts = [clean_tweet(t[0]) for t in tweets_data]
labels = [t[1] for t in tweets_data]

# --- Vectorisation TF-IDF ---
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(texts).toarray()
y = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)

# --- Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

# --- Réseau de neurones ---
class SimpleNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return self.sigmoid(x)

model = SimpleNN(input_dim=1000)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# --- Entraînement ---
for epoch in range(5):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    print(f"Époque {epoch+1}, perte: {loss.item():.4f}")

# --- Évaluation ---
model.eval()
with torch.no_grad():
    preds = model(X_test_tensor)
    pred_labels = (preds > 0.5).int()
    true_labels = y_test.int()
    print(classification_report(true_labels, pred_labels))

# --- Prédiction personnalisée ---
def predict_sentiment(tweet):
    cleaned = clean_tweet(tweet)
    vec = vectorizer.transform([cleaned]).toarray()
    vec_tensor = torch.tensor(vec, dtype=torch.float32)
    with torch.no_grad():
        pred = model(vec_tensor)
        return "positif" if pred.item() > 0.5 else "négatif"

if __name__ == "__main__":
    print("Exemple de prédiction :")
    print(predict_sentiment("shut the fuck up"))
    print(predict_sentiment("je suis trop heureux aujourd'hui !"))
