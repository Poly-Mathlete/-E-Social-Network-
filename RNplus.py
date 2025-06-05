import re
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


def clean_tweet(tweet):
    tweet = tweet.lower()
    tweet = re.sub(r"http\S+", "", tweet)
    tweet = re.sub(r"@\w+", "", tweet)
    tweet = re.sub(r"#\w+", "", tweet)
    tweet = re.sub(r"rt\s?:", "", tweet)
    tweet = re.sub(r"[^\w\s]", "", tweet)
    tweet = re.sub(r"\s+", " ", tweet).strip()
    return tweet

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

texts = [t[0] for t in tweets_data if isinstance(t[0], str) and len(t[0].strip()) > 0]
labels = [t[1] for t in tweets_data if isinstance(t[0], str) and len(t[0].strip()) > 0]


vectorizer = TfidfVectorizer(max_features=1024)
X = vectorizer.fit_transform(texts).toarray()
y = torch.tensor(labels).float()


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
X_train = torch.tensor(X_train).float()
X_test = torch.tensor(X_test).float()
y_train = y_train.view(-1, 1).float()
y_test = y_test.view(-1, 1).float()


class SentimentNN(nn.Module):# c'est ici que ça change
    def __init__(self):
        super(SentimentNN, self).__init__()
        self.fc1 = nn.Linear(1024, 512)
        

        self.relu1 = nn.LeakyReLU(0.01)# ici on utilise LeakyReLU pour éviter le problème de vanishing gradient( c'est une activation qui permet de garder une petite pente pour les valeurs négatives)
        self.drop1 = nn.Dropout(0.3)# # le dropout sert contre le sur-apprentissage
        self.fc2 = nn.Linear(512, 128)
        
        self.relu2 = nn.LeakyReLU(0.01)
        self.drop2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.drop1(self.relu1(self.fc1(x)))
        x = self.drop2(self.relu2(self.fc2(x)))

        x = self.fc3(x)
        return x

model = SentimentNN()

# et la on vas gérer le déséquilibre des classes pour que le modèle n'apprenne pas à "favoriser" la classe majoritaire.
n_pos = y_train.sum()
n_neg = len(y_train) - n_pos
pos_weight = torch.tensor([n_neg / n_pos]).float() # e ratio permet de dire : “Un exemple positif compte autant que X exemples négatifs.”
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)#fonction de perte adaptée aux classes déséquilibrées

optimizer = torch.optim.AdamW(model.parameters(), lr=0.005)#AdamW a meilleure convergence (plus rapide et plus stable),moins d’overfitting

# --- Entraînement ---
print("⚡ Entraînement du modèle...")
for epoch in range(20):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}/20 - Loss: {loss.item():.4f}")

# --- Évaluation ---
model.eval()
with torch.no_grad():
    logits = model(X_test)
    probs = torch.sigmoid(logits)
    y_pred_label = (probs > 0.49).int()
    print("\n✅ Résultats sur le test :")
    print(classification_report(y_test.int(), y_pred_label))

    # Matrice de confusion
    cm = confusion_matrix(y_test.int(), y_pred_label)
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
        prob = torch.sigmoid(output).item()
    return "positif" if prob >= 0.5 else "négatif"

if __name__ == "__main__":
    print("Exemple :", predict_sentiment("I love how this works"))
    