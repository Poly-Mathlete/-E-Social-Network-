import re
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# la ça change pcq tout ce qui posait problème dans le code précédent a été corrigé
def clean_tweet(tweet): #
    tweet = tweet.lower()
    tweet = re.sub(r"http\S+", "", tweet)
    tweet = re.sub(r"@\w+", "", tweet)
    tweet = re.sub(r"#\w+", "", tweet)
    tweet = re.sub(r"rt\s?:", "", tweet)
    tweet = re.sub(r"[^\w\s]", "", tweet)
    tweet = re.sub(r"\s+", " ", tweet).strip()
    return tweet

# la c'est pareil
tweets_data = []
with open("MMM.csv", "r", encoding="latin1") as f:
    reader = csv.reader(f)
    for row in reader:
        if len(row) >= 6:
            label = 0 if row[0] == "0" else 1 if row[0] == "4" else None
            if label is not None:
                tweets_data.append((clean_tweet(row[5]), label))

texts = [t[0] for t in tweets_data]
labels = [t[1] for t in tweets_data]

# ici aussi
vectorizer = TfidfVectorizer(max_features=1024)
X = vectorizer.fit_transform(texts).toarray()
y = torch.tensor(labels).float().view(-1, 1)

# 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42) #
X_train, X_test = torch.tensor(X_train).float(), torch.tensor(X_test).float()
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
#Il découpe automatiquement le dataset en mini-lots (batches) de 64 exemples.Il mélange les exemples à chaque epoch pour éviter que le modèle n’apprenne un ordre fixe.


class SentimentNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1024, 512)
        self.relu1 = nn.LeakyReLU(0.01)
        self.drop1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(512, 128)
        self.relu2 = nn.LeakyReLU(0.01)
        self.drop2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.drop1(self.relu1(self.fc1(x)))
        x = self.drop2(self.relu2(self.fc2(x)))
        return self.fc3(x)

model = SentimentNN()
n_pos = y_train.sum()
n_neg = len(y_train) - n_pos
pos_weight = torch.tensor([n_neg / n_pos]).float()
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.AdamW(model.parameters(), lr=0.01)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
#ca vas nous permettre de réduire le taux d'apprentissage de moitié tous les 5 epochs, ce qui est utile pour affiner l'entraînement du modèle.

# 
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

plt.plot(losses)
plt.title("Courbe de perte")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()


model.eval()
with torch.no_grad():
    logits = model(X_test)
    probs = torch.sigmoid(logits)

#la le but vas être de tester un seuil optimal pour la classification pour trouver le meilleur F1-score.
best_f1, best_thresh = 0, 0.5
for t in [i / 100 for i in range(48, 52)]:
    preds = (probs > t).int()
    f1 = f1_score(y_test.int(), preds)
    if f1 > best_f1:
        best_f1 = f1
        best_thresh = t

final_preds = (probs > best_thresh).int()
print(f"\n✅ Meilleur seuil : {best_thresh:.2f} | F1-score : {best_f1:.4f}")
print(classification_report(y_test.int(), final_preds))

cm = confusion_matrix(y_test.int(), final_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["négatif", "positif"])
disp.plot(cmap=plt.cm.Blues)
plt.title("Matrice de confusion")
plt.show()# permet de visualiser la matrice de confusion


def predict_sentiment(tweet):
    cleaned = clean_tweet(tweet)
    vec = vectorizer.transform([cleaned]).toarray()
    with torch.no_grad():
        output = model(torch.tensor(vec).float())
        prob = torch.sigmoid(output).item()
    return "positif" if prob >= best_thresh else "négatif"

if __name__ == "__main__":
    print("Exemple :", predict_sentiment("I love how this works"))
