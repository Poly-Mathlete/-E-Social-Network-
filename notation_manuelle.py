import random

# Charger les fichiers
with open("scored_tweets_lexical_refined1.txt", "r", encoding="utf-8") as f:
    scored_lines = [line.strip() for line in f if line.strip()]

with open("CorpusRandomCleaned/cleaned_tweets1.txt", "r", encoding="utf-8") as f:
    cleaned_lines = [line.strip() for line in f if line.strip()]

# Analyser les scores et extraire ID et score
tweet_scores = {}
for line in scored_lines:
    try:
        parts = line.split(" | ")
        # Extrait l'ID
        id_part = parts[0]  # "Tweet ID: 1"
        tweet_id = int(id_part.split(":")[1].strip())
        # Extrait le score
        score_part = parts[1]  # "Score: 0.30"
        score = float(score_part.split(":")[1].strip())
        tweet_scores[tweet_id] = score
    except Exception as e:
        print(f"Erreur parsing ligne : {line}\n{e}")

# Fonction pour transformer un score numérique en polarité
def interpret_score(score):
    if score > 0:
        return "positif"
    elif score < 0:
        return "négatif"
    else:
        return "neutre"

# Boucle d’évaluation manuelle
total = 0
correct = 0

print("Tape 'bon' si le score te paraît correct, 'pas bon' sinon. Tape 'stop' pour arrêter.\n")

while True:
    idx = random.choice(list(tweet_scores.keys()))
    score = tweet_scores[idx]
    sentiment = interpret_score(score)

    # Attention à la correspondance indice tweet/ligne : ici on suppose que tweet_id correspond à l'indice dans cleaned_lines (0-based ?)
    # Si Tweet ID commence à 1, on fait idx-1
    tweet_text = cleaned_lines[idx - 1] if 0 < idx <= len(cleaned_lines) else "[Tweet non trouvé]"

    print(f"Tweet : {tweet_text}")
    print(f"Score détecté : {sentiment} (valeur brute : {score})")
    user_input = input("→ Ton avis ('bon' / 'pas bon' / 'stop') : ").strip().lower()

    if user_input == "stop":
        break
    elif user_input == "bon":
        correct += 1
        total += 1
    elif user_input == "pas bon":
        total += 1
    else:
        print("Réponse invalide. Réessaye.")

# Résultat final
if total > 0:
    normalized_score = correct / total
    print(f"\n Score validé pour {correct} / {total} évaluations.")
    print(f" Score normalisé : {normalized_score:.2f}")
else:
    print("\nAucune évaluation effectuée.")