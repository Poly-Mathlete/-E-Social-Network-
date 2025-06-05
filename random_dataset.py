import pandas as pd

# Définir le chemin vers le fichier CSV complet
input_file = "training.1600000.processed.noemoticon.csv"

# Lire uniquement les colonnes utiles, ici on suppose que la première colonne est le label
# et que le fichier n'a pas d'en-tête
df = pd.read_csv(input_file, encoding="ISO-8859-1", header=None)

# Les colonnes sont par défaut : 0 = label, 5 = texte du tweet
df.columns = ["label", "id", "date", "query", "user", "text"]

# Sélectionner les tweets avec label 0 (négatif) et 4 (positif)
neg_tweets = df[df["label"] == 0]
pos_tweets = df[df["label"] == 4]

# Échantillonnage aléatoire de 100 000 lignes dans chaque catégorie
neg_sample = neg_tweets.sample(n=100_000, random_state=42)
pos_sample = pos_tweets.sample(n=100_000, random_state=42)

# Fusionner et mélanger les deux ensembles
balanced_sample = pd.concat([neg_sample, pos_sample]).sample(frac=1, random_state=42)

# Sauvegarder dans un nouveau fichier
balanced_sample.to_csv("balanced_tweets_200k.csv", index=False, encoding="utf-8")
