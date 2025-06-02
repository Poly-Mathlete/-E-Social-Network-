import os
print("Current working directory:", os.getcwd())

from gensim.models import KeyedVectors
import re
import numpy as np
import matplotlib.pyplot as plt

# Charger le modèle FastText français
model_path = "cc.fr.300.vec.gz"  # Mets le chemin correct si besoin
print("Chargement du modèle FastText français...")
model = KeyedVectors.load_word2vec_format(model_path, binary=False)
print("Modèle chargé.")

def clean_and_tokenize(text):
    text = text.lower()
    return re.findall(r'\b\w+\b', text)

def average_vector(words, model):
    vectors = [model[word] for word in words if word in model]
    if not vectors:
        return np.zeros(model.vector_size)
    return np.mean(vectors, axis=0)

from numpy.linalg import norm

def cosine_similarity(vec1, vec2):
    if norm(vec1) == 0 or norm(vec2) == 0:
        return 0.0
    return np.dot(vec1, vec2) / (norm(vec1) * norm(vec2))

def sentiment_score(tweet, model, pos_vocab, neg_vocab):
    words = clean_and_tokenize(tweet)
    tweet_vec = average_vector(words, model)
    pos_vec = average_vector(pos_vocab, model)
    neg_vec = average_vector(neg_vocab, model)

    sim_pos = cosine_similarity(tweet_vec, pos_vec)
    sim_neg = cosine_similarity(tweet_vec, neg_vec)

    score = max(0, min(10, (sim_pos - sim_neg) * 10 + 5))
    return round(score, 2), sim_pos, sim_neg

all_scores = []

for i in range(3,5):
    source_file = f"CorpusRandomCleaned/cleaned_tweets{i}.txt"
    output_file = f"Corpus_scores_vec/cores_tweets_vec{i}.txt"

    with open(source_file, encoding="utf-8") as f:
        tweets = f.read().splitlines()

    # Chargement des vocabulaires positifs et négatifs (en français !)
    with open("positive_vocab_1000.txt", encoding="utf-8") as f:
        positive_vocab = [line.strip() for line in f if line.strip()]
    
    with open("negative_vocab_1000.txt", encoding="utf-8") as f:
        negative_vocab = [line.strip() for line in f if line.strip()]

    results = []
    for tweet in tweets:
        if not tweet.strip():
            continue
        score, pos_score, neg_score = sentiment_score(tweet, model, positive_vocab, negative_vocab)
        all_scores.append(score)
        results.append(f" | Score: {score}, Pos: {pos_score}, Neg: {neg_score}")

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(results))


import gensim.downloader as api


# Charger un modèle anglais (GloVe ou FastText)
print("Chargement du modèle anglais...")
model = api.load("glove-wiki-gigaword-100")  # ou "fasttext-wiki-news-subwords-300"
print("Modèle chargé.")

def clean_and_tokenize(text):
    text = text.lower()
    return re.findall(r'\b\w+\b', text)

def average_vector(words, model):
    vectors = [model[word] for word in words if word in model]
    if not vectors:
        return np.zeros(model.vector_size)
    return np.mean(vectors, axis=0)

from numpy.linalg import norm

def cosine_similarity(vec1, vec2):
    if norm(vec1) == 0 or norm(vec2) == 0:
        return 0.0
    return np.dot(vec1, vec2) / (norm(vec1) * norm(vec2))

def sentiment_score(tweet, model, pos_vocab, neg_vocab):
    words = clean_and_tokenize(tweet)
    tweet_vec = average_vector(words, model)
    pos_vec = average_vector(pos_vocab, model)
    neg_vec = average_vector(neg_vocab, model)

    sim_pos = cosine_similarity(tweet_vec, pos_vec)
    sim_neg = cosine_similarity(tweet_vec, neg_vec)

    score = max(0, min(10, (sim_pos - sim_neg) * 10 + 5))
    return round(score, 2), sim_pos, sim_neg

all_scores = []

for i in range(1, 3):
    source_file = f"CorpusRandomCleaned/cleaned_tweets{i}.txt"
    output_file = f"Corpus_scores_vec/cores_tweets_vec{i}.txt"

    with open(source_file, encoding="utf-8") as f:
        tweets = f.read().splitlines()

    # Chargement des vocabulaires positifs et négatifs (en anglais !)
    with open("positive_vocab_1000.txt", encoding="utf-8") as f:
        positive_vocab = [line.strip() for line in f if line.strip()]
    
    with open("negative_vocab_1000.txt", encoding="utf-8") as f:
        negative_vocab = [line.strip() for line in f if line.strip()]

    results = []
    for tweet in tweets:
        if not tweet.strip():
            continue
        score, pos_score, neg_score = sentiment_score(tweet, model, positive_vocab, negative_vocab)
        all_scores.append(score)
        results.append(f" | Score: {score}, Pos: {pos_score}, Neg: {neg_score}")

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(results))

# Visualisation de la distribution des scores
plt.hist(all_scores, bins=20)
plt.title("Distribution of Sentiment Scores")
plt.xlabel("Score")
plt.ylabel("Number of Tweets")
plt.show()


# Visualisation de la distribution des scores
plt.hist(all_scores, bins=20)
plt.title("Distribution des scores de sentiment")
plt.xlabel("Score")
plt.ylabel("Nombre de tweets")
plt.show()