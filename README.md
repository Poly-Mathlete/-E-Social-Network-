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


#### 2. Approche lexicale (`approche_lexicale.py` ce code se trouve dans la branche Myriam)

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

Pour passer à la deuxième phase du projet (entraînement de modèles plus complexes pour prédire le sentiment d’un tweet automatiquement), nous avons utilisé une base de données existante de 1,6 million de tweets étiquetés automatiquement.

- Cette base a été réduite à 200 000 tweets pour des raisons de performance (mémoire, vitesse d'entraînement).
- Cette réduction aléatoire a été réalisée via le script `random_dataset.py`.
- Le fichier final `balanced_tweets_200k.csv` est équilibré (même nombre de tweets positifs et négatifs) et servira pour l’apprentissage supervisé.



### 1. Régression linéaire naïve (`reg_lin_improved.py`)

Cette première approche, volontairement simple, applique une régression linéaire directement sur les scores TF-IDF des tweets.  
Le modèle est entraîné pour prédire un score numérique (positif ou négatif), ensuite transformé en classe (`positif` si > 0, `négatif` sinon).

Objectif : servir de baseline très simple pour comparer avec des modèles plus avancés.



### 2. Régression logistique (présente dans `models_EI.ipynb`)

La régression logistique est un modèle supervisé standard pour les tâches de classification binaire.  
Dans le notebook, nous l'entraînons sur les vecteurs TF-IDF pour prédire les sentiments. Elle offre un bon compromis entre performance et interprétabilité, et permet d’obtenir une probabilité d’appartenance à chaque classe.


### 3. Random Forest (présente dans `models_EI.ipynb`)

La Random Forest est un ensemble de plusieurs arbres de décision.  
Elle est entraînée sur les mêmes vecteurs TF-IDF. Ce modèle :
- apprend des règles de décision complexes,
- offre de bonnes performances sans beaucoup de réglages.
Elle permet également d’évaluer l’importance relative des mots dans la prédiction.


### 4. Réseau de neurones (`RNfast.py`)

Ce modèle utilise un réseau de neurones profond (multi-layer perceptron) avec plusieurs couches :
- une couche d'entrée prenant les vecteurs TF-IDF,
- une ou plusieurs couches cachées avec activation ReLU,
- une couche de sortie avec fonction `sigmoïde` ou `softmax`.

Le modèle est entraîné avec un optimiseur (`Adam`) pour classer les tweets.  
Ce réseau permet d’explorer des représentations non linéaires complexes, avec des résultats parfois meilleurs que les modèles classiques.


Tous les modèles ont été évalués à l’aide d’une métrique de précision sur un jeu de test séparé (20 % des données).  
Les performances varient selon la complexité du modèle et la capacité à généraliser à des tweets variés.










