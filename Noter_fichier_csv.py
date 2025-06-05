import pandas as pd  # bibliothèque pour lire et manipuler fichier csv
import random as rd  # bibliothèque pour faire des choix aléatoires

# Lire le fichier CSV contenant les tweets
# encoding="latin-1" permet d'ouvrir correctement les caractères spéciaux
# header=None signifie que le fichier n'a pas de ligne de titre
df = pd.read_csv("balanced_tweets_200k.csv", encoding="latin-1", header=None)

# On donne des noms aux colonnes du tableau pour y accéder plus facilement
df.columns = ["polarity", "id", "date", "query", "user", "text"]

# On garde uniquement les tweets qui ont une polarité de 0 (négatif) ou 4 (positif)
df = df[df["polarity"].isin([0, 4])]

# On prend un échantillon aléatoire de 5000 tweets pour l'évaluation
# reset_index(drop=True) réinitialise les numéros de lignes
df = df.sample(5000).reset_index(drop=True)

# Fonction pour afficher la polarité sous forme de texte lisible
def interpret_polarity(p):
    return "négatif" if p == 0 else "positif"

# Initialisation des compteurs qui servent à quantifier à quel point je suis d'accord avec le scorring 
total = 0    # nombre total d’évaluations effectuées
correct = 0  # nombre d’évaluations jugées correctes par l’utilisateur

# Instructions pour l’utilisateur
print("Tape 'bon' si tu es d'accord avec la polarité, 'pas bon' sinon. Tape 'stop' pour arrêter.\n")

# Boucle d’évaluation des tweets
while True:
    # Sélectionne un tweet au hasard parmi les 5000
    row = df.sample(1).iloc[0]

    # Récupère le texte du tweet et sa polarité interprétée en texte
    tweet = row["text"]
    sentiment = interpret_polarity(row["polarity"])

    # Affiche le tweet et la polarité détectée automatiquement
    print(f"Tweet : {tweet}")
    print(f"Polarité détectée : {sentiment}")

    # Demande à l’utilisateur de donner son avis
    user_input = input("→ Ton avis ('bon' / 'pas bon' / 'stop') : ").strip().lower()

    # Si l'utilisateur veut arrêter l'évaluation
    if user_input == "stop":
        break
    # Si l'utilisateur est d'accord avec l’étiquette automatique
    elif user_input == "bon":
        correct += 1
        total += 1
    # Si l'utilisateur pense que la polarité est incorrecte
    elif user_input == "pas bon":
        total += 1
    # Si l'utilisateur tape autre chose
    else:
        print("Réponse invalide. Réessaye.")

# Après la boucle, afficher les résultats
if total > 0:
    # Calcul du score : proportion de bonnes réponses
    normalized_score = correct / total
    print(f"\n Score validé pour {correct} / {total} évaluations.")
    print(f" Score normalisé : {normalized_score:.2f}")
else:
    # Si l’utilisateur n’a rien évalué
    print("\nAucune évaluation effectuée.")