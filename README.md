# -E-Social-Network-  
**Projet Mazars** : Analyse de sentiments sur des données X/Twitter via un pipeline de Machine Learning complet, de la collecte à la visualisation, pour évaluer l'e-réputation.


## Partie 1 : Étiquetage automatique de tweets

L'objectif de cette première partie est de construire une base annotée à partir de 4000 tweets bruts, en utilisant des méthodes automatiques pour évaluer le sentiment (positif / négatif) de chaque tweet.



### Prétraitement des tweets

Deux scripts de nettoyage ont été développés :

- `clean_Final.py` : réalise un nettoyage classique (ponctuation, stopwords, emojis...) sans tenir compte de la négation.
- `clean_with_neg.py` : même logique, mais conserve les mots de négation (ex. `pas`, `jamais`) afin d'améliorer les performances des approches lexicales.

---

### Méthodes d’étiquetage automatique

#### 1. Modèle probabiliste (`score_eval_model_prob.py`)

Cette méthode utilise l'approche TF-IDF, elle applique ce modèle aux tweets nettoyés pour prédire un score de sentiment, transformé ensuite en étiquette (`positif` ou `négatif`).


#### 2. Approche lexicale (`approche_lexicale.py`)

Cette méthode repose sur un lexique de mots positifs et négatifs.  
Chaque tweet est analysé en comptant les occurrences de mots à polarité positive ou négative :

- Si le score net est positif → le tweet est classé `positif`.
- Sinon → `négatif`.

Cependant, cette approche dépend fortement de la richesse du lexique utilisé. Un lexique trop court donne de mauvaises performances, notamment sur les tweets avec du langage familier ou des structures complexes.

Pour évaluer les performances de l’approche lexicale, le script (`notation_manuelle.py`) permet de :

- tirer aléatoirement 50 tweets,
- les annoter manuellement,
- comparer ces annotations avec celles générées automatiquement,
- afficher une précision.

---

#### 3. Modèle vectoriel (`score_eval_model_vec.py`)

Cette méthode repose sur des vecteurs de mots pré-entraînés (fastText). 
Un vecteur moyen est calculé pour le tweet en moyennant les vecteurs FastText des mots qu’il contient. De la même manière, un vecteur moyen est calculé pour un vocabulaire positif et un vocabulaire négatif préalablement définis. La similarité cosinus est ensuite calculée entre le vecteur du tweet et chacun des deux vecteurs de référence (positif et négatif). Un score final de sentiment est ensuite calculé pour prédire l’étiquette (`positif` ou `négatif`).



#### Évaluation vectorielle (`noter_model_vectoriel.py`)

Même principe que pour l’approche lexicale :

- sélection aléatoire de tweets,
- étiquetage manuel,
- comparaison avec le modèle,
- mesure de performance.



## Constitution d’un dataset d’entraînement plus large

Pour passer à la deuxième phase du projet (entraînement de modèles plus complexes), nous avons utilisé une base de données existante de 1,6 million de tweets étiquetés automatiquement.

- Cette base a été réduite à 200 000 tweets pour des raisons de performance (mémoire, vitesse d'entraînement).
- Cette réduction aléatoire a été réalisée via le script `random_dataset.py`.
- Le fichier final `balanced_tweets_200k.csv` est équilibré (même nombre de tweets positifs et négatifs) et servira pour l’apprentissage supervisé.







