import re
from pathlib import Path
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import TreebankWordTokenizer

# Télécharger les ressources NLTK 
try:
    stopwords.words("french")
except LookupError:
    nltk.download("stopwords", quiet=True)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download("punkt", quiet=True)


# Initialisation du tokenizer
tokenizer = TreebankWordTokenizer()

# Préparation des regex 
regex_artifacts_sources = re.compile(
    r'\|'
    r'\[Saut de retour à la ligne\]|'
    r'(?:<ed><U\+[0-9A-F]{4}><U\+[0-9A-F]{4}>)+|'
    r'<U\+[0-9A-F]{4,6}>|'
    r'<ed>'
)
regex_usernames = re.compile(r'@\w+')
regex_urls = re.compile(r'https?://\S+')
regex_hashtags = re.compile(r'#\w+')
regex_apostrophes = re.compile(r"\b\w+'") # Pour l', j', c' etc.
regex_quotes = re.compile(r'"')
regex_punctuation = re.compile(r'[.,;:!?()\[\]{}\\/|`~^<>«»=]')


# Définir les mots de négation à conserver
negation_words_fr = {"ne", "pas", "jamais", "plus", "rien", "aucun", "aucune", "n'"} # "n'" pour "n'est", "n'a" etc.
negation_words_en = {
    "no", "not", "never", "don't", "isn't", "aren't", "wasn't", "weren't",
    "haven't", "hasn't", "hadn't", "couldn't", "wouldn't", "shouldn't",
    "mightn't", "mustn't", "shan't", "needn't", "ain't", "nor"
} # TreebankWordTokenizer sépare "don't" en "do" et "n't", donc "n't" n'est pas nécessaire ici, mais d'autres contractions oui.
# Pour être plus précis TreebankWordTokenizer donne: "don't" -> ["do", "n't"]. "n't" est un token.
# La regex_apostrophes r"\b\w+'" ne cible pas "n't" mais "l'", "j'".
# Pour le français "n'est" -> ["n'", "est"]. Donc "n'" est utile.

# Pré-calculer les listes de stopwords modifiées
base_stopwords_fr = set(stopwords.words("french"))
custom_stopwords_fr = base_stopwords_fr - negation_words_fr

base_stopwords_en = set(stopwords.words("english"))
custom_stopwords_en = base_stopwords_en - negation_words_en
# Ajouter 'nt' car TreebankWordTokenizer le sépare de "not" dans les contractions
custom_stopwords_en.discard("n't") # S'assurer que "n't" n'est PAS un stopword

# Traitement des fichiers
for i in range(1, 5): 
    # Définir le chemin du fichier source à lire
    source_file = f"CorpusRandomTwitter/randomtweets{i}.txt"
    
    # Créer le dossier de sortie s'il n'existe pas
    output_dir = Path("CorpusRandomCleaned")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Chemin du fichier nettoyé à sauvegarder
    dest_file = output_dir / f"cleaned_tweets_neg_aware{i}.txt"

    cleaned_tweets = []  # Liste pour stocker les tweets nettoyés
    
    # Déterminer la langue (les fichiers 1 et 2 sont en français, les autres en anglais)
    langue = "french" if i in [1, 2] else "english"

    # Choisir la liste de stopwords personnalisée selon la langue
    if langue == "french":
        stop_words_to_use = custom_stopwords_fr
    else:
        stop_words_to_use = custom_stopwords_en

    # Lecture du fichier source
    try:
        with open(source_file, encoding="utf-8") as f:
            lines = f.read().splitlines()
    except FileNotFoundError:
        print(f"Fichier {source_file} non trouvé. Passe au suivant.")
        continue  # Passe au fichier suivant si le fichier courant n'existe pas

    # Traitement ligne par ligne
    for line_num, line in enumerate(lines):
        # Ignorer la première ligne d’en-tête ou les lignes vides
        if line == '"","x"' or not line.strip():
            if line_num == 0 and line == '"","x"':
                 continue
            elif not line.strip():
                continue

        # Extraire l'ID et le texte du tweet via regex
        m = re.match(r'^"(\d+)","(.*)"$', line)
        if not m:
            # Ignorer les lignes qui ne correspondent pas au format attendu
            continue

        tweet_id, tweet_text = m.groups()

        # Nettoyage du texte :
        # Supprimer les artefacts visuels, mentions, URLs, hashtags, etc.
        t = regex_artifacts_sources.sub('', tweet_text)
        t = t.replace('""', '"')
        t = regex_usernames.sub('', t)
        t = regex_urls.sub('', t)
        t = regex_hashtags.sub('', t)

        # Suppression des contractions comme l', d', j', etc. (mais on préserve les négations comme "n'")
        t = regex_apostrophes.sub('', t)

        # Supprimer les guillemets typographiques et autres ponctuations
        t = regex_quotes.sub('', t)
        t = t.replace("'", "")  # Supprimer les apostrophes simples restantes
        t = regex_punctuation.sub(' ', t)  # Remplacer la ponctuation par des espaces

        # Mise en minuscule et nettoyage des espaces superflus
        t = t.lower()
        t = re.sub(r'\s{2,}', ' ', t).strip()

        # Tokenisation du texte (avec TreebankWordTokenizer par ex.)
        tokens = tokenizer.tokenize(t)

        # Filtrer les stopwords tout en conservant certains tokens comme "rt"
        filtered_tokens = [
            word.upper() if word.lower() == 'rt' else word
            for word in tokens
            if word.lower() not in stop_words_to_use or word.lower() == 'rt'
        ]

        # Ne conserver que les tweets qui ont encore du contenu après nettoyage
        if filtered_tokens:
            cleaned_tweets.append(f"{tweet_id} : {filtered_tokens}")

    # Écriture du fichier nettoyé
    with open(dest_file, "w", encoding="utf-8") as f:
        f.write("\n".join(cleaned_tweets))

    print(f"Fichier nettoyé sauvegardé : {dest_file}")