import random
import importlib.util
import ast
import csv

# === 1. Import dynamique du fichier score_eval_model_vec.py ===
spec = importlib.util.spec_from_file_location("score_module", "score_eval_model_vec.py")
score_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(score_module)

# === 2. Charger les vocabulaires ===
with open("positive_vocab_1000.txt", encoding="utf-8") as f:
    positive_vocab = [line.strip() for line in f if line.strip()]

with open("negative_vocab_1000.txt", encoding="utf-8") as f:
    negative_vocab = [line.strip() for line in f if line.strip()]

# === 3. Charger les tweets nettoyés (sous forme de liste de mots) ===
cleaned_tweet_files = [
    f"CorpusRandomCleaned_2/cleaned_tweets{i}.txt" for i in range(1, 5)
]

all_tweets = []

for path in cleaned_tweet_files:
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                # Transforme la chaîne représentant une liste en vraie liste de mots
                tokens = ast.literal_eval(line)
                tweet = " ".join(tokens)  # Reconstitue une phrase
                all_tweets.append(tweet)
            except Exception as e:
                print(f"⚠️ Erreur dans {path} : {e}")

# === 4. Charger les tweets bruts (format CSV texte) ===
random_tweet_files = [
    f"CorpusRandomTwitter/randomtweets{i}.txt" for i in range(1, 5)
]

for path in random_tweet_files:
    try:
        with open(path, encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)  # ignore l'en-tête
            all_tweets += [row[1].strip() for row in reader if len(row) > 1 and row[1].strip()]
    except FileNotFoundError:
        print(f"⚠️ Fichier non trouvé : {path}")

# === 5. Interface utilisateur ===
print("=== Interface de test des scores ===")

oui_count = 0
total_questions = 0

while True:
    tweet = random.choice(all_tweets)
    print("\nTweet :")
    print(tweet)

    # Calcul du score
    score, pos_score, neg_score = score_module.sentiment_score(
        tweet, score_module.model, positive_vocab, negative_vocab
    )

    print(f"\nScore du modèle : {score} (Pos: {pos_score:.2f}, Neg: {neg_score:.2f})")

    user_input = input("Ce score te paraît-il correct ? (oui / non / stop): ").strip().lower()

    if user_input == "stop":
        if total_questions == 0:
            print("Aucune réponse donnée. Fin de session.")
        else:
            normalized_score = oui_count / total_questions
            print(f"\n=== Résumé ===\nNombre total de réponses : {total_questions}")
            print(f"Nombre de 'oui' : {oui_count}")
            print(f"Score normalisé (taux de 'oui') : {normalized_score:.2f}")
        break
    elif user_input == "oui":
        oui_count += 1
        total_questions += 1
        print("→ Tu es d’accord avec le modèle.\n")
    elif user_input == "non":
        total_questions += 1
        print("→ Tu n’es pas d’accord avec le modèle.\n")
    else:
        print("Réponse non reconnue. Tape 'oui', 'non' ou 'stop'.\n")