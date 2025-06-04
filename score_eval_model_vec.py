import os
print("Current working directory:", os.getcwd())

from gensim.models import KeyedVectors
import re
import numpy as np
import matplotlib.pyplot as plt

# Charger le modèle FastText français
model_path = "cc.fr.300.vec.gz"  # Mets le chemin correct si besoin
# print("Chargement du modèle FastText français...")
# model = KeyedVectors.load_word2vec_format(model_path, binary=False)
# print("Modèle chargé.")

# def clean_and_tokenize(text):
#     text = text.lower()
#     return re.findall(r'\b\w+\b', text)

# def average_vector(words, model):
#     vectors = [model[word] for word in words if word in model]
#     if not vectors:
#         return np.zeros(model.vector_size)
#     return np.mean(vectors, axis=0)

# from numpy.linalg import norm

# def cosine_similarity(vec1, vec2):
#     if norm(vec1) == 0 or norm(vec2) == 0:
#         return 0.0
#     return np.dot(vec1, vec2) / (norm(vec1) * norm(vec2))

# def sentiment_score(tweet, model, pos_vocab, neg_vocab):
#     words = clean_and_tokenize(tweet)
#     tweet_vec = average_vector(words, model)
#     pos_vec = average_vector(pos_vocab, model)
#     neg_vec = average_vector(neg_vocab, model)

#     sim_pos = cosine_similarity(tweet_vec, pos_vec)
#     sim_neg = cosine_similarity(tweet_vec, neg_vec)

#     score = max(0, min(10, (sim_pos - sim_neg) * 10 + 5))
#     return round(score, 2), sim_pos, sim_neg

# all_scores = []

# for i in range(3,5):
#     source_file = f"CorpusRandomCleaned_2/cleaned_tweets{i}.txt"
#     output_file = f"Corpus_scores_vec/cores_tweets_vec{i}.txt"

#     with open(source_file, encoding="utf-8") as f:
#         tweets = f.read().splitlines()

#     # Chargement des vocabulaires positifs et négatifs (en français !)
#     with open("positive_vocab_1000.txt", encoding="utf-8") as f:
#         positive_vocab = [line.strip() for line in f if line.strip()]
    
#     with open("negative_vocab_1000.txt", encoding="utf-8") as f:
#         negative_vocab = [line.strip() for line in f if line.strip()]

#     results = []
#     for tweet in tweets:
#         if not tweet.strip():
#             continue
#         score, pos_score, neg_score = sentiment_score(tweet, model, positive_vocab, negative_vocab)
#         all_scores.append(score)
#         results.append(f"{tweet}: {score}")

#     with open(output_file, "w", encoding="utf-8") as f:
#         f.write("\n".join(results))


import gensim.downloader as api


# Charger un modèle anglais 
print("Chargement du modèle anglais...")
model = api.load("glove-wiki-gigaword-100")  # ou "fasttext-wiki-news-subwords-300"
print("Modèle chargé.")

def clean_and_tokenize(text):
    text = text.lower()
    return re.findall(r'\b\w+\b', text)
from gensim.models.phrases import Phrases, Phraser

# Collect all tokenized tweets for bigram training
all_tokenized_tweets = []
for i in range(1, 5):
    source_file = f"CorpusRandomCleaned_2/cleaned_tweets{i}.txt"
    with open(source_file, encoding="utf-8") as f:
        tweets = f.read().splitlines()
    for tweet in tweets:
        tokens = clean_and_tokenize(tweet)
        all_tokenized_tweets.append(tokens)

# Train bigram model
bigram = Phrases(all_tokenized_tweets, min_count=5, threshold=10)
bigram_mod = Phraser(bigram)

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

    score = max(0, min(10, (sim_pos - sim_neg) * 5 + 1))
    score =  max(-1.0, min(1.0, score-1))  # Normaliser entre -1 et 1

    return round(score, 2), sim_pos, sim_neg

all_scores = []

for i in range(3,5):
    source_file = f"CorpusRandomCleaned_2/cleaned_tweets{i}.txt"
    output_file = f"Corpus_scores_vec_enhanced/cores_tweets_vec{i}.txt"

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
        results.append(f" {tweet}: {score}")

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(results))

# Visualisation de la distribution des scores
# plt.hist(all_scores, bins=20)
# plt.title("Distribution of Sentiment Scores")
# plt.xlabel("Score")
# plt.ylabel("Number of Tweets")
# plt.show()


# # Visualisation de la distribution des scores
# plt.hist(all_scores, bins=20)
# plt.title("Distribution des scores de sentiment")
# plt.xlabel("Score")
# plt.ylabel("Nombre de tweets")
# plt.show()

# # Comptage des catégories de scores
# count_high = sum(1 for s in all_scores if s > 6)
# count_low = sum(1 for s in all_scores if s < 4.5)
# count_mid = sum(1 for s in all_scores if 4.5 <= s <= 6)

# labels = ['Score > 6', 'Score < 4.5', '4.5 ≤ Score ≤ 6']
# sizes = [count_high, count_low, count_mid]
# colors = ['#66b3ff', '#ff9999', '#99ff99']

# plt.figure(figsize=(6,6))
# plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=140)
# plt.title("Répartition des scores de sentiment")
# plt.axis('equal')
# plt.show()

# import matplotlib.pyplot as plt
# import numpy as np # Should already be imported in your script

# def plot_sentiment_distribution_pie_enhanced(scores_list, title="Répartition des scores de sentiment"):
#     """
#     Generates a more sophisticated pie chart and a donut chart for sentiment score distribution.

#     Args:
#         scores_list (list): A list of sentiment scores.
#         title (str): The title for the chart.
#     """
#     if not scores_list:
#         print("No scores provided to plot. Skipping pie chart generation.")
#         return

#     # Define categories, their conditions, labels for legend, and colors
#     # This order (Negative, Neutral, Positive) is often standard for visualization
#     categories_config = [
#         {'name': 'Négatif', 'condition': lambda s: s < 4.5, 'label': 'Score < 4.5 (Négatif)', 'color': '#ff9999'},  # Light Red
#         {'name': 'Neutre', 'condition': lambda s: 4.5 <= s <= 6, 'label': '4.5 ≤ Score ≤ 6 (Neutre)', 'color': '#ffcc99'}, # Light Orange/Yellow
#         {'name': 'Positif', 'condition': lambda s: s > 6, 'label': 'Score > 6 (Positif)', 'color': '#99ff99'}    # Light Green
#     ]
#     # If you prefer the color scheme from your image (Blue for Pos, Green for Neu, Red for Neg):
#     # categories_config = [
#     #     {'name': 'Positif', 'condition': lambda s: s > 6, 'label': 'Score > 6 (Positif)', 'color': '#66b3ff'},      # Light Blue
#     #     {'name': 'Neutre', 'condition': lambda s: 4.5 <= s <= 6, 'label': '4.5 ≤ Score ≤ 6 (Neutre)', 'color': '#99ff99'}, # Light Green
#     #     {'name': 'Négatif', 'condition': lambda s: s < 4.5, 'label': 'Score < 4.5 (Négatif)', 'color': '#ff9999'}  # Light Red
#     # ]


#     counts = np.array([sum(1 for s in scores_list if cat['condition'](s)) for cat in categories_config])
    
#     # Filter out categories with zero counts for plotting
#     plot_labels = []
#     plot_sizes = []
#     plot_colors = []
#     plot_explode = []

#     # Determine which slice to explode (e.g., the largest non-zero slice)
#     # We'll calculate explode factors based on original categories, then filter
#     explode_factors = [0.0] * len(counts)
#     if sum(c > 0 for c in counts) > 1 : # Only explode if there's more than one visible slice
#         non_zero_counts = [c for c in counts if c > 0]
#         if non_zero_counts:
#             max_val = max(non_zero_counts)
#             # Find first index of max_val in original counts array
#             max_idx = -1
#             for i_expl, c_expl in enumerate(counts):
#                 if c_expl == max_val:
#                     max_idx = i_expl
#                     break
#             if max_idx != -1:
#                  explode_factors[max_idx] = 0.05 # Explode the largest slice slightly


#     for i, count in enumerate(counts):
#         if count > 0:
#             plot_labels.append(categories_config[i]['label'])
#             plot_sizes.append(count)
#             plot_colors.append(categories_config[i]['color'])
#             plot_explode.append(explode_factors[i])


#     if not plot_sizes:
#         print("All score categories are empty after filtering. Nothing to plot.")
#         return

#     # --- Enhanced Pie Chart ---
#     fig1, ax1 = plt.subplots(figsize=(10, 7)) # Adjusted figure size for legend

#     def autopct_format(pct_value):
#         """Format autopct to show percentage and count."""
#         absolute_value = int(round(pct_value / 100. * np.sum(plot_sizes)))
#         return f"{pct_value:.1f}%\n({absolute_value})"

#     wedges, texts, autotexts = ax1.pie(
#         plot_sizes,
#         explode=plot_explode,
#         labels=None,  # We use a legend instead of direct labels on slices
#         colors=plot_colors,
#         autopct=autopct_format,
#         shadow=True,
#         startangle=140,
#         pctdistance=0.75, # Adjust as needed, closer to center if showing counts
#         wedgeprops={'edgecolor': 'gray', 'linewidth': 0.5}
#     )

#     # Improve styling of percentage text
#     plt.setp(autotexts, size=9, weight="bold", color="black")
#     for autotext in autotexts: # Add a slight background to text for readability
#          autotext.set_bbox(dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1))


#     ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
#     ax1.set_title(title, fontsize=16, pad=20)

#     # Add a legend
#     ax1.legend(wedges, plot_labels,
#                title="Catégories des Scores",
#                loc="center left",
#                bbox_to_anchor=(1, 0, 0.5, 1), # Position legend outside the pie
#                fontsize=10)

#     plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust rect to make space for legend if bbox_to_anchor is used
#     plt.show()

#     # --- Donut Chart ---
#     fig2, ax2 = plt.subplots(figsize=(10, 7))

#     wedges_donut, texts_donut, autotexts_donut = ax2.pie(
#         plot_sizes,
#         explode=plot_explode, # Explode can also be used with donut
#         labels=None,
#         colors=plot_colors,
#         autopct=autopct_format,
#         startangle=140,
#         pctdistance=0.80, # For donut, text can be further from center
#         wedgeprops=dict(width=0.4, edgecolor='white', linewidth=1) # This makes it a donut!
#     )
    
#     plt.setp(autotexts_donut, size=9, weight="bold", color="black")
#     for autotext_donut in autotexts_donut:
#          autotext_donut.set_bbox(dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1))

#     ax2.axis('equal')
#     ax2.set_title(f"{title} (Donut Chart)", fontsize=16, pad=20)
#     ax2.legend(wedges_donut, plot_labels,
#                title="Catégories des Scores",
#                loc="center left",
#                bbox_to_anchor=(1, 0, 0.5, 1),
#                fontsize=10)
    
#     plt.tight_layout(rect=[0, 0, 0.85, 1])
#     plt.show()



# # Appel de la fonction avec les scores collectés
# if 'all_scores' in locals() and all_scores: # Check if all_scores exists and is not empty
#     plot_sentiment_distribution_pie_enhanced(all_scores, title="Répartition des Scores de Sentiment")
# else:
#     print("Variable 'all_scores' or 'all_scores_collected' not found or is empty. Cannot generate pie chart.")

import csv

# Pour stocker tous les scores de tweets
all_tweet_scores = []

# Pour stocker les scores de chaque token
all_token_scores = []

# for i in range(1, 5):
#     source_file = f"CorpusRandomCleaned_2/cleaned_tweets{i}.txt"
#     with open(source_file, encoding="utf-8") as f:
#         tweets = f.read().splitlines()

#     # Chargement des vocabulaires positifs et négatifs
#     with open("positive_vocab_1000.txt", encoding="utf-8") as f:
#         positive_vocab = [line.strip() for line in f if line.strip()]
#     with open("negative_vocab_1000.txt", encoding="utf-8") as f:
#         negative_vocab = [line.strip() for line in f if line.strip()]

#     for tweet in tweets:
#         if not tweet.strip():
#             continue
#         score, pos_score, neg_score = sentiment_score(" ".join(tokens_with_bigrams), model, positive_vocab, negative_vocab)
#         all_tweet_scores.append([score])

#         # Score de chaque token
#         tokens = clean_and_tokenize(tweet)
#         tokens = clean_and_tokenize(tweet)
#     tokens_with_bigrams = bigram_mod[tokens]

#     for token in tokens_with_bigrams:
#         token_score, _, _ = sentiment_score(token, model, positive_vocab, negative_vocab)
#         all_token_scores.append([token, token_score])

# # Écriture du CSV des scores de tweets
# with open("all_tweet_scores.csv", "w", newline='', encoding="utf-8") as f:
#     writer = csv.writer(f)
#     writer.writerow(["score"])
#     writer.writerows(all_tweet_scores)

# # Écriture du CSV des scores de tokens
# with open("all_token_scores.csv", "w", newline='', encoding="utf-8") as f:
#     writer = csv.writer(f)
#     writer.writerow(["token", "score"])
#     writer.writerows(all_token_scores)


