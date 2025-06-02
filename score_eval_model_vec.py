import os
print("Current working directory:", os.getcwd())

import gensim.downloader as api
model = api.load("glove-wiki-gigaword-100")
import re

def clean_and_tokenize(text):
    text = text.lower()
    return re.findall(r'\b\w+\b', text)

import numpy as np

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

    score = (sim_pos - sim_neg + 1) * 5  # normalisé entre 0 et 10
    return round(score, 2), sim_pos, sim_neg

for i in range(1,5):
    source_file = f"CorpusRandomCleaned/cleaned_tweets{i}.txt"
    output_file = f"Corpus_scores_vec/cores_tweets_vec{i}.txt"

    with open(source_file, encoding="utf-8") as f:
        tweets = f.read().splitlines()

    # Chargement des vocabulaires positifs et négatifs
    with open("positive_vocab_1000.txt", encoding="utf-8") as f:
        positive_vocab = [line.strip() for line in f if line.strip()]
    
    with open("negative_vocab_1000.txt", encoding="utf-8") as f:
        negative_vocab = [line.strip() for line in f if line.strip()]

    results = []
    for tweet in tweets:
        if not tweet.strip():
            continue
        score, pos_score, neg_score = sentiment_score(tweet, model, positive_vocab, negative_vocab)
        results.append(f" | Score: {score}, Pos: {pos_score}, Neg: {neg_score}")

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(results))