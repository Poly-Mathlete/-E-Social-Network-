import math
import re
from collections import Counter
def file_content_to_list(file_path):
    """
    Lit un fichier texte et transforme son contenu (un mot/phrase par ligne)
    en une liste Python.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        lines = content.strip().split('\n')
        word_list = [line.strip() for line in lines if line.strip()]
        return word_list
    except FileNotFoundError:
        print(f"Erreur : Le fichier {file_path} n'a pas été trouvé.")
        return []



# Exemples simples
positive_vocab = file_content_to_list(positive_vocab_file := "positive_vocab_1000.txt")

negative_vocab = file_content_to_list(negative_vocab_file := "negative_vocab_1000.txt")


idf_scores = {word: math.log(100 / (1 + i)) for i, word in enumerate(set(positive_vocab + negative_vocab))}

def clean_and_tokenize(tweet):
    tweet = tweet.lower()
    return re.findall(r'\b\w+\b', tweet)

def score_tf_idf(tweet_words, vocab, idf_dict):
    counter = Counter(tweet_words)
    score = 0.0
    for word in vocab:
        if word in counter:
            tf = counter[word]
            idf = idf_dict.get(word, 1.0)
            score += math.log(1 + tf) * idf
    return score

def compute_sentiment_score(tweet):
    words = clean_and_tokenize(tweet)
    pos_score = score_tf_idf(words, positive_vocab, idf_scores)
    neg_score = score_tf_idf(words, negative_vocab, idf_scores)
    epsilon = 1e-5
    final_score = (pos_score / (pos_score + neg_score + epsilon)) * 10
    return round(final_score, 2), pos_score, neg_score



for i in range(1, 5):
    source_file = f"CorpusRandomCleaned_2/cleaned_tweets{i}.txt"
    output_file = f"Corpus_scores/sentiment_scores_tweets{i}.txt"

    with open(source_file, encoding="utf-8") as f:
        tweets = f.read().splitlines()

    results = []
    for tweet in tweets:
        if not tweet.strip():
            continue
        score, pos_score, neg_score = compute_sentiment_score(tweet)
        results.append(f" | Score: {score}, Pos: {pos_score}, Neg: {neg_score}")

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(results))




#Les tweets sont très courts, donc peu de mots détectables.

# Beaucoup de mots sont hors vocabulaire (fautes, emojis, nouvelles expressions).

# IDF peut écraser les scores si les mots ne sont pas assez fréquents dans l’ensemble

