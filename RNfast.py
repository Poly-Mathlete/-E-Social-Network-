import re
import csv
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score
from torch.utils.data import TensorDataset, DataLoader

#Ici le code rapide mais avec les avancés du modèle précédent
def clean_tweet(tweet):
    tweet = tweet.lower()
    tweet = re.sub(r"http\S+|@\w+|#\w+|rt\s?:", "", tweet)
    tweet = re.sub(r"[^\w\s]", "", tweet)
    return re.sub(r"\s+", " ", tweet).strip()

# --- Chargement des données ---
texts, labels = [], []
with open("MMM.csv", "r", encoding="latin1") as f:
    reader = csv.reader(f)
    for row in reader:
        if len(row) >= 6 and row[0] in ["0", "4"]:
            label = 0 if row[0] == "0" else 1
            texts.append(clean_tweet(row[5]))
            labels.append(label)

# vecteur leger
vectorizer = TfidfVectorizer(max_features=256)
X = vectorizer.fit_transform(texts).toarray()
y = torch.tensor(labels).float().view(-1, 1)

#comme avant on sépare les données
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_test = torch.tensor(X_train).float(), torch.tensor(X_test).float()
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=128, shuffle=True)

#model simplifié et rapide
class FastSentiment(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(256, 64)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.2)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.drop(self.relu(self.fc1(x)))
        return self.fc2(x)

model = FastSentiment()


n_pos = y_train.sum()
n_neg = len(y_train) - n_pos
pos_weight = torch.tensor([n_neg / n_pos]).float()
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)


optimizer = optim.AdamW(model.parameters(), lr=0.01)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)


losses = []
for epoch in range(20):
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    scheduler.step()
    losses.append(total_loss / len(train_loader))
    print(f"Epoch {epoch+1} - Loss: {losses[-1]:.4f}")

#print des différentes metriques
plt.plot(losses)
plt.title("Courbe de perte")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()


model.eval()
with torch.no_grad():
    probs = torch.sigmoid(model(X_test))

# --- Recherche du meilleur seuil ---
best_f1, best_thresh = 0, 0.5
f1_scores = []
thresholds = [i / 100 for i in range(40, 61)]
for t in thresholds:
    preds = (probs > t).int()
    f1 = f1_score(y_test.int(), preds)
    f1_scores.append(f1)
    if f1 > best_f1:
        best_f1 = f1
        best_thresh = t

#affichage des résultats
plt.plot(thresholds, f1_scores)
plt.title("F1-score selon le seuil")
plt.xlabel("Seuil")
plt.ylabel("F1-score")
plt.show()

#rendu du meilleur seuil et de la classification
final_preds = (probs > best_thresh).int()
print(f"\n Meilleur seuil : {best_thresh:.2f} | F1-score : {best_f1:.4f}")
print(classification_report(y_test.int(), final_preds))


cm = confusion_matrix(y_test.int(), final_preds)
print("\nMatrice de confusion :\n", cm)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["négatif", "positif"])
disp.plot(cmap=plt.cm.Blues)
plt.title("Matrice de confusion")
plt.show()


def predict_sentiment(tweet):
    cleaned = clean_tweet(tweet)
    vec = vectorizer.transform([cleaned]).toarray()
    with torch.no_grad():
        prob = torch.sigmoid(model(torch.tensor(vec).float())).item()
    return "positif" if prob >= best_thresh else "négatif"

if __name__ == "__main__":
    print("Exemple :", predict_sentiment("I love how this works"))
