import re
from pathlib import Path


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
        # Supprime les espaces multiples
        t = re.sub(r'\s{2,}', ' ', t).strip()

        cleaned_tweets.append(f"{tweet_num} : {t}")
    # Sauvegarde du fichier nettoyé
    with open(dest_file, "w", encoding="utf-8") as f:
        f.write("\n".join(cleaned_tweets))