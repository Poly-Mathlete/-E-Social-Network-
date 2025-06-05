import re
from pathlib import Path

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import TreebankWordTokenizer

# Télécharger les ressources NLTK
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)

# Initialisation du tokenizer (plus robuste que word_tokenize)
tokenizer = TreebankWordTokenizer()

# Préparation des regex
# on supprime les artefacts typiques des tweets encodés de manière étrange
regex_artifacts_sources = re.compile(
    r'\|'
    r'\[Saut de retour à la ligne\]|'
    r'(?:<ed><U\+[0-9A-F]{4}><U\+[0-9A-F]{4}>)+|'
    r'<U\+[0-9A-F]{4,6}>|'
    r'<ed>'
)

# on supprime les noms d'utilisateurs mentionnés 
regex_usernames = re.compile(r'@\w+')

# on supprime les URL (http ou https)
regex_urls = re.compile(r'https?://\S+')

# on supprime les hashtags uniquement
regex_hashtags = re.compile(r'#\w+')

# on supprime les contractions comme l', j', c', etc.
regex_apostrophes = re.compile(r"\b\w+'")

# on supprime les guillemets doubles
regex_quotes = re.compile(r'"')

# on supprime la ponctuation générale
regex_punctuation = re.compile(r'[.,;:!?()\[\]{}\\/|`~^<>«»=]')

# Traitement des fichiers 
# Boucle sur 4 fichiers nommés randomtweets1.txt à randomtweets4.txt
for i in range(1, 5):
    source_file = f"CorpusRandomTwitter/randomtweets{i}.txt"
    
    # Crée le dossier de sortie s’il n’existe pas
    output_dir = Path("CorpusRandomCleaned")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Définition du chemin de sortie pour les tweets nettoyés
    dest_file = output_dir / f"cleaned_tweets{i}.txt"

    cleaned_tweets = []

    # Choix de la langue pour les stopwords
    langue = "french" if i in [1, 2] else "english"
    stop_words = set(stopwords.words(langue))

    with open(source_file, encoding="utf-8") as f:
        lines = f.read().splitlines()

    for line in lines:
        # Ignore les lignes vides ou invalides
        if line == '"","x"' or not line.strip():
            continue
        # Extraction du numéro de tweet et du texte via regex
        m = re.match(r'^"(\d+)","(.*)"$', line)
        if not m:
            continue
        tweet_num, tweet_text = m.groups()

        # Nettoyage du texte 
        # Supprime les artefacts liés à l'encodage
        t = regex_artifacts_sources.sub('', tweet_text)
        # Corrige les doubles guillemets ("" → ")
        t = t.replace('""', '"')
        # Supprime les noms d’utilisateurs
        t = regex_usernames.sub('', t)
        # Supprime les liens
        t = regex_urls.sub('', t)
        # Supprime les hashtags
        t = regex_hashtags.sub('', t)
        # Supprime les contractions du type l', j', etc.
        t = regex_apostrophes.sub('', t)
        # Supprime tous les guillemets
        t = regex_quotes.sub('', t)
        # Supprime les apostrophes restantes
        t = t.replace("'", "")
        # Supprime la ponctuation
        t = regex_punctuation.sub(' ', t)
        # Met en minuscule
        t = t.lower()
        # Supprime les espaces multiples
        t = re.sub(r'\s{2,}', ' ', t).strip()

        # Tokenisation
        tokens = tokenizer.tokenize(t)

        # Conservation de RT en majuscule même après passage en minuscule
        filtered_tokens = [
            word.upper() if word.lower() == 'rt' else word
            for word in tokens
            if word.lower() not in stop_words or word.lower() == 'rt'
        ]

        # On garde le tweet sous forme de liste de mots
        cleaned_tweets.append(f"{tweet_num} : {filtered_tokens}")
        
    # Sauvegarde du fichier nettoyé
    with open(dest_file, "w", encoding="utf-8") as f:
        f.write("\n".join(cleaned_tweets))