import re
from pathlib import Path

# Préparation des regex
regex_artifacts_sources = re.compile(
    r'\|'
    r'\[Saut de retour à la ligne\]|'
    r'(?:<ed><U\+[0-9A-F]{4}><U\+[0-9A-F]{4}>)+|'
    r'<U\+[0-9A-F]{4,6}>|'
    r'<ed>'
)
regex_usernames = re.compile(r'@\w+')

for i in range(1, 5):
    source_file = f"CorpusRandomTwitter/randomtweets{i}.txt"
    output_dir = Path("CorpusRandomCleaned")
    output_dir.mkdir(parents=True, exist_ok=True)
    dest_file = output_dir / f"cleaned_tweets{i}.txt"

    cleaned_tweets = []
    with open(source_file, encoding="utf-8") as f:
        lines = f.read().splitlines()

    for line in lines:
        if line == '"","x"' or not line.strip():
            continue
        m = re.match(r'^"(\d+)","(.*)"$', line)
        if not m:
            continue
        tweet_num, tweet_text = m.groups()

        # nettoyage
        t = regex_artifacts_sources.sub('', tweet_text)
        t = t.replace('""', '"')
        t = regex_usernames.sub('', t)
        t = re.sub(r'\s{2,}', ' ', t).strip()

        cleaned_tweets.append(f"{tweet_num} : {t}")

    with open(dest_file, "w", encoding="utf-8") as f:
        f.write("\n".join(cleaned_tweets))
