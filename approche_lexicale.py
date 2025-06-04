import re
from pathlib import Path
import ast # Pour convertir la chaîne de la liste en une vraie liste Python

# --- DÉFINITIONS DES LEXIQUES ET MOTS CLÉS ---

# 1. Lexique de sentiments (mis à jour avec votre liste et mes estimations de scores)
sentiment_lexicon_fr = {
    # Positifs / Enthousiasme
    "excellent": 0.9, "superbe": 0.8, "merveilleux": 0.85, "adore": 0.7,
    "bon": 0.6, "bien": 0.5, "aime": 0.65, "heureux": 0.7, "positif": 0.75,
    "intéressant": 0.4, "sympa": 0.5, "cool": 0.6, "génial": 0.85, "passionnant": 0.7, # "cool" un peu plus haut
    "plaisir": 0.6, "rire": 0.5, "sourire": 0.6, "joie": 0.7, "fidèle": 0.5,
    "stylé": 0.6, "ouf": 0.7, "lourd": 0.4, # "lourd" peut être ambigu, mais souvent positif en argot
    "fort": 0.5, "respect": 0.6, # "respect" plus fort
    "grave": 0.3, # "grave" comme adverbe d'intensité (positif ou négatif selon contexte, ici neutre-positif)
    "de ouf": 0.7, "archi": 0.3, "abusé": 0.2, # "abusé" peut être ++ ou --, ici légèrement positif
    "validé": 0.7, "amoureux": 0.8, "amour" : 0.8,
    "top": 0.8, "ouais": 0.2, "kiff": 0.7, "mdr": 0.4, "ptdr": 0.5, "lol": 0.4,
    "bg": 0.6, "sérieux": 0.1, # "sérieux" peut être neutre ou exclamatif
    "wow": 0.7, "trop bien": 0.85, "nickel": 0.7,
    "incroyable": 0.8, "impressionnant": 0.7, "dingue": 0.6, # "dingue" peut être ++
    "inspirant": 0.75, "mignon": 0.65, "peace": 0.7, "chance": 0.6, "victoire": 0.9,
    "trop fort": 0.8, "magnifique": 0.9, "j’adore": 0.75, "je kiffe": 0.75, # Doublon avec adore/kiff
    "belle": 0.7, "bon délire": 0.65, "confort": 0.4, "motive": 0.6,
    "tranquille": 0.3, "stylax": 0.6, "meilleur": 0.9, "aimer": 0.65, # Doublon
    "nintendo": 0.3, # Marque, peut être contextuel
    "c’est d’la balle": 0.8, "wsh c’est bien": 0.6, "on est là": 0.3,
    "gentil": 0.6, "cimer": 0.5, # Argot pour merci
    "talentueux": 0.75, "adorable": 0.7, "waouh": 0.7, # Doublon
    "vive": 0.6, "coolos": 0.6, "hâte": 0.5, "oklm": 0.4,
    "ultra stylé": 0.8, "succès": 0.8,
    "doux": 0.4, "drôle": 0.6, "fun": 0.7, "chill": 0.5,
    "intelligent": 0.7, "goood": 0.6, # Orthographe
    "amusant": 0.6, "coeur": 0.5, # Souvent dans des expressions positives
    "préféré": 0.7, "bravo": 0.8,
    "courageux": 0.7, "sympathique": 0.6, "chanceux": 0.6, "topissime": 0.9,
    "gagné": 0.8, "d'accord": 0.2, "yes": 0.7, "kiffe": 0.7, # Doublon
    "trop cool": 0.8, "excitant": 0.7, "giga top": 0.9, "yay": 0.7,
    "plage": 0.5, "réussi": 0.8,
    "jpp": 0.1, # "j'en peux plus (de rire)" = positif, "j'en peux plus (d'exaspération)" = négatif. Score neutre/légèrement positif par défaut ici.
    "stars": 0.3, "félicitations": 0.8, "content": 0.7,
    "détente": 0.5, "vacances": 0.7, "super": 0.8, "parfait": 0.95,
    "juste wow": 0.75, "dar": 0.7, # Argot
    "beau": 0.7, "c’est grave bien": 0.8, "détendu": 0.4, "excellentissime": 0.95,
    "repos": 0.4, "impatient": 0.3, # Peut être négatif si l'attente est mauvaise
    "ambiance": 0.2, "cadeau": 0.5,
    "confiance": 0.6, "let’s go": 0.5,

    # Neutres / faibles (j'ai déplacé jpp ici avec score neutre par défaut)
    "ok": 0.1, "moyen": 0.0, "neutre": 0.0, "correct": 0.2,
    "jcrois": 0.0, "genre": 0.0,

    # Négatifs / Péjoratifs
    "tristesse": -0.8, "angoisse": -0.7, "haine": -0.9, "ah ouais non": -0.5,
    "mdrrrr": -0.2, # Peut être ironique/négatif, ou juste rire. Contexte crucial. Ici légèrement négatif si dans une liste négative.
    "énervé": -0.6, "planté": -0.5, "saoule": -0.6,
    "abusé": -0.4, # Contexte négatif: "c'est abusé"
    "c’est relou": -0.7, "galère": -0.6, "zarma": -0.1, # Moquerie
    "trahison": -0.8, "fail": -0.7, "toxique": -0.7, "crise": -0.6,
    "malheureux": -0.8, "trop chiant": -0.75, "raté": -0.6,
    "panique": -0.7, "dégueu": -0.8, "dégueulasse": -0.85, # Doublon
    "chelou": -0.5, "pire": -0.9, "c’est claqué": -0.8, "blacklist": -0.5,
    "beurk": -0.7, "stop": -0.2, "agacé": -0.5, "non merci": -0.4,
    "non mais allo": -0.5, "naze": -0.7, "ennuyé": -0.5,
    "zéro": -0.8, "relou": -0.7, # Doublon
    "blasant": -0.6, "wtf": -0.4, "non": -0.3, # 'non' seul est moins fort qu'une négation structurée
    "on est foutus": -0.9, "WTF": -0.4, # Doublon
    "trop nul": -0.85, "dégoût total": -0.9, "mauvais": -0.6,
    "cancel": -0.5, "terrible": -0.9, "problème": -0.5, "nul": -0.8, # Doublon
    "j’ai peur": -0.7, "voleur": -0.6, "boloss": -0.7, "perdu": -0.5,
    "chiant": -0.7, # Doublon
    "nul à chier": -0.95, "chagrin": -0.7, "blessé": -0.6,
    "con": -0.8, "danger": -0.7, "échec": -0.7, "aucune chance": -0.8,
    "fait chier": -0.8, "bordel": -0.5, # Interjection, peut être neutre ou exprimer frustration
    "déception": -0.7, "arnaque": -0.8, "risque": -0.4, "triste": -0.8, # Doublon
    "galère de ouf": -0.7, "flemme": -0.4, "bête": -0.5, "c’est nul": -0.8,
    "déprime": -0.7, "moche": -0.6, "la haine": -0.9, "rage": -0.7,
    "jpp d’eux": -0.6, # "j'en peux plus d'eux"
    "peur": -0.7, # Doublon
    "vénère": -0.7, "je meurs": -0.3, # Souvent hyperbolique et pas littéralement négatif. Peut être lié à "jpp".
    "dégoût": -0.8, "seum": -0.7, "pute": -0.9, "furieux": -0.8,
    "rien à foutre": -0.5, "déçu": -0.7, # Doublon
    "wtffff": -0.4, # Doublon
    "frustration": -0.6, "horrib": -0.9, # Orthographe
    "merde": -0.7, "pleurer": -0.6, "las": -0.4, "défaite": -0.8,
    "horrible": -0.95, "colère": -0.8, "pff": -0.3, "c’est mort": -0.8,
    "fatigué": -0.4, "gros fail": -0.8, "foutu": -0.7, "stupide": -0.7,
    "affreux": -0.85, "catastrophe": -0.9, "craint": -0.6, "putain": -0.6, # Interjection
    "cramé": -0.5, "t’as vu ça": 0.0, # Neutre, exclamation de surprise
    "difficulté": -0.4, "pas ouf": -0.6, "sans espoir": -0.8, "deg": -0.7,
    "jamais": -0.3, # Si utilisé seul comme interjection "Jamais !"
    "ptn": -0.5, # Interjection

    # Mots à surveiller (modérateurs ou contextuels) - certains déjà listés plus haut
    "austérité": -0.5, "ennuyeux": -0.5, "ennui": -0.4,
    "haineux": -0.8, "mort": -0.7, # Peut être positif dans "mort de rire"

    # RT : score neutre
    "rt": 0.0
}
# 2. Mots de négation
negation_words = {"ne", "n'", "n’", "pas", "jamais", "plus", "rien", "aucun", "aucune", "non"}
NEGATION_WINDOW = 3 # Fenêtre d'influence de la négation

# 3. Intensificateurs et Atténuateurs
intensifiers = {
    "très": 1.5, "extrêmement": 2.0, "vraiment": 1.3, "tellement": 1.4, 
    "beaucoup": 1.2, "incroyablement": 1.7, "fortement": 1.4, "trop": 1.3,
    "totalement": 1.6, "complètement": 1.5, "sacrément": 1.4
}
diminishers = {
    "peu": 0.5, "un peu": 0.6, "plutôt": 0.7, "assez": 0.8, 
    "moins": 0.7, "légèrement": 0.75, "guère": 0.4, "à peine": 0.5
}

# 4. Conjonctions contrastives
contrast_conjunctions = {"mais", "cependant", "pourtant", "néanmoins", "toutefois", "malgré", "quoique", "bien que"}
CONTRAST_FACTOR = 1.5 # Facteur de pondération pour la clause après la conjonction

# --- FONCTION D'ANALYSE DE SENTIMENT LEXICALE AMÉLIORÉE ---
def analyze_sentiment_lexical_refined(tokens):
    """
    Analyse le sentiment d'une liste de tokens avec gestion de la négation,
    des modificateurs et des contrastes.
    Les tokens sont supposés être en minuscules et nettoyés.
    """
    if not tokens:
        return 0.0, "neutre", 0

    score_clauses = []
    current_clause_score = 0.0
    current_clause_sentiment_word_count = 0
    
    negation_influence_countdown = 0
    modifier_factor = 1.0
    
    active_clause_has_sentiment_word = False # Pour suivre si la clause actuelle a des mots de sentiment

    for i, token in enumerate(tokens):
        # La sortie de clean_mimi.py peut avoir 'RT' en majuscules.
        # On s'assure de travailler avec des minuscules pour la recherche dans les lexiques.
        token_lower = token.lower() 

        # Ignorer "RT" pour le calcul de score s'il est encore là
        if token_lower == 'rt':
            continue

        word_score = 0.0
        is_sentiment_bearing_word = False

        # 1. Vérifier si c'est un intensificateur ou un atténuateur
        if token_lower in intensifiers:
            modifier_factor = intensifiers[token_lower]
            continue 
        elif token_lower in diminishers:
            modifier_factor = diminishers[token_lower]
            continue

        # 2. Vérifier si c'est une conjonction contrastive
        if token_lower in contrast_conjunctions:
            score_clauses.append(current_clause_score)
            current_clause_score = 0.0
            current_clause_sentiment_word_count = 0 # Réinitialiser pour la nouvelle clause
            active_clause_has_sentiment_word = False
            # Réinitialiser les modificateurs et la négation au changement de clause
            modifier_factor = 1.0
            negation_influence_countdown = 0
            continue

        # 3. Obtenir le score du lexique
        if token_lower in sentiment_lexicon_fr:
            word_score = sentiment_lexicon_fr[token_lower]
            is_sentiment_bearing_word = True
        
        if is_sentiment_bearing_word:
            active_clause_has_sentiment_word = True
            current_clause_sentiment_word_count += 1

            # 4. Appliquer le modificateur
            word_score *= modifier_factor
            modifier_factor = 1.0 # Réinitialiser pour le prochain mot

            # 5. Appliquer l'influence de la négation
            if negation_influence_countdown > 0:
                word_score *= -1
        
        current_clause_score += word_score
        
        # 6. Gérer la négation pour les mots suivants
        if token_lower in negation_words:
            negation_influence_countdown = NEGATION_WINDOW 
        elif negation_influence_countdown > 0 : 
            # Décrémenter la fenêtre de négation pour chaque mot (sentimental ou non)
            # qui n'est pas lui-même un mot de négation.
            # Si un mot sentimental est trouvé, il consomme la négation.
            # S'il n'y a pas de mot sentimental dans la fenêtre, la négation "s'épuise".
            negation_influence_countdown -= 1


    # Ajouter le score de la dernière clause traitée
    score_clauses.append(current_clause_score)
    
    # 7. Agréger les scores des clauses
    final_score = 0.0
    total_sentiment_words_in_scored_clauses = 0

    if not score_clauses:
        final_score = 0.0
    elif len(score_clauses) == 1:
        final_score = score_clauses[0]
        # Le compte de mots de sentiment est celui de la seule clause
        # (déjà stocké dans current_clause_sentiment_word_count, mais celui-ci est pour la *dernière* clause)
        # Pour être précis, il faudrait sommer les current_clause_sentiment_word_count de chaque clause
        # si on les stockait séparément.
        # Pour l'instant, on va se baser sur le compte global si une seule clause.
        # Si la clause était vide de mots de sentiment, current_clause_sentiment_word_count serait 0.
        # On a besoin du compte total de mots de sentiment sur TOUTES les clauses pour la normalisation.
        # Faisons un compte global plus simple pour la normalisation :
        num_sentiment_words_overall = sum(1 for t in tokens if t.lower() in sentiment_lexicon_fr)

    else: # Plus d'une clause, gestion du contraste
        # Appliquer le CONTRAST_FACTOR à la dernière clause
        # (celle après la conjonction contrastive)
        score_clauses[-1] *= CONTRAST_FACTOR
        final_score = sum(score_clauses) # Somme simple après pondération de la dernière clause
        num_sentiment_words_overall = sum(1 for t in tokens if t.lower() in sentiment_lexicon_fr)


    # Normalisation optionnelle (pour un score moyen par mot de sentiment)
    # if num_sentiment_words_overall > 0:
    #    normalized_score = final_score / num_sentiment_words_overall
    # else:
    #    normalized_score = 0.0
    # Pour cet exemple, on utilise final_score (somme, potentiellement pondérée par contraste)

    # Étiquetage
    # Ajustez ces seuils en fonction de la distribution des scores que vous obtenez
    # et de la granularité souhaitée.
    threshold_pos = 0.1 # Exemple
    threshold_neg = -0.1 # Exemple

    if final_score > threshold_pos:
        label = "positif"
    elif final_score < threshold_neg:
        label = "négatif"
    else:
        label = "neutre"
        
    return final_score, label, num_sentiment_words_overall


# --- TRAITEMENT DU FICHIER DE TWEETS NETTOYÉS ---
# Doit correspondre à la sortie de clean_mimi.py
# Format attendu: chaque ligne est "ID : ['token1', 'token2', ...]"
# (ou le nom de fichier que vous avez utilisé, ex: cleaned_tweets1.txt)
input_file_path = Path("CorpusRandomCleaned/cleaned_tweets_neg_aware1.txt")
output_file_path = Path("scored_tweets_lexical_refined1.txt")

if not input_file_path.exists():
    print(f"Fichier d'entrée {input_file_path} non trouvé.")
else:
    results_output = []
    print(f"Traitement du fichier : {input_file_path}")
    with open(input_file_path, "r", encoding="utf-8") as f_in:
        for line_num, line_content in enumerate(f_in):
            line_content = line_content.strip()
            if not line_content:
                continue

            match = re.match(r"(\d+)\s*:\s*(.+)", line_content) # '+' pour capturer la liste
            if match:
                tweet_id_str = match.group(1)
                tokens_str = match.group(2) # La chaîne représentant la liste
                
                try:
                    # ast.literal_eval est plus sûr pour convertir une chaîne en structure Python (liste ici)
                    tokens_list = ast.literal_eval(tokens_str)
                    if not isinstance(tokens_list, list): # S'assurer que c'est bien une liste
                        print(f"Ligne {line_num+1}: Le contenu parsé n'est pas une liste pour tweet {tweet_id_str}. Contenu: {tokens_str[:70]}...")
                        continue
                except (ValueError, SyntaxError) as e:
                    print(f"Ligne {line_num+1}: Erreur parsing tokens (ast.literal_eval) pour tweet {tweet_id_str}: {tokens_str[:70]}... - {e}")
                    # Fallback si ast.literal_eval échoue (par ex. si ce n'est pas une liste valide)
                    # Ceci est une rustine et indique un problème dans le format d'entrée
                    # ou dans la façon dont les tokens ont été sauvegardés.
                    # Votre clean_mimi.py devrait produire une liste Python valide sous forme de chaîne.
                    # tokens_list = [t.strip().strip("'\"") for t in tokens_str.strip("[]").split(',') if t.strip()]
                    continue # Préférable de sauter la ligne si le format est incorrect
                                
                score, label, sentiment_word_count = analyze_sentiment_lexical_refined(tokens_list)
                results_output.append(f"Tweet ID: {tweet_id_str} | Score: {score:.2f} | Label: {label} | Mots Sentiment: {sentiment_word_count} | Tokens: {tokens_list}")
            else:
                print(f"Ligne {line_num+1}: Format non reconnu (attendu 'ID : [tokens]'). Ligne ignorée: {line_content[:70]}...")

    with open(output_file_path, "w", encoding="utf-8") as f_out:
        for res_line in results_output:
            f_out.write(res_line + "\n")
            
    print(f"Traitement Raffiné terminé. Résultats dans : {output_file_path}")

    