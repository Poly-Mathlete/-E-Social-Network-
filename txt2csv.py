import re
import csv

def transform_txt_to_csv_flexible(input_txt_file, output_csv_file):
    """
    Transforms lines from a text file into a CSV file with ID and Score.
    Handles optional "RT :" after the initial ID and separator.

    Input format examples:
    "24 : role of that... : 0.35"
    "2 : RT : Varun Dhawan... : 6.26"

    Output CSV format per line:
    "ID,Score" (e.g., "24,0.35")
    """
    # Regex ajustée :
    # - ^\s*(\d+)\s* : Capture l'ID au début (groupe 1)
    # - : : Correspond au premier séparateur ':'
    # - \s*(?:RT\s*:\s*)? : Groupe non-capturant (?:...) pour "RT : " optionnel
    #     - RT : Les lettres RT
    #     - \s*:\s* : Un autre séparateur ":" après RT, avec des espaces optionnels
    #     - ? : Rend tout le groupe "RT : " optionnel
    # - .*? : Le texte du tweet (non gourmand)
    # - :\s* : Le dernier séparateur ":" avant le score
    # - ([-+]?\d*\.?\d+) : Capture le score (groupe 2)
    # - \s*$ : Fin de ligne
    line_pattern = re.compile(r"^\s*(\d+)\s*:\s*(?:RT\s*:\s*)?.*:\s*([-+]?\d*\.?\d+)\s*$")

    extracted_data = []

    try:
        with open(input_txt_file, 'r', encoding='utf-8') as infile:
            for line_number, line_content in enumerate(infile, 1): # Ajout du numéro de ligne pour le débogage
                line = line_content.strip()
                if not line:
                    continue

                match = line_pattern.match(line)
                if match:
                    tweet_id = match.group(1)
                    score = match.group(2) # Le score est maintenant le groupe 2
                    extracted_data.append([tweet_id, score])
                else:
                    print(f"Attention (ligne {line_number}): La ligne n'a pas pu être parsée : '{line[:100]}...'")

    except FileNotFoundError:
        print(f"Erreur : Le fichier d'entrée '{input_txt_file}' n'a pas été trouvé.")
        return
    except Exception as e:
        print(f"Une erreur est survenue lors de la lecture du fichier : {e}")
        return

    if not extracted_data:
        print("Aucune donnée n'a été extraite. Le fichier CSV ne sera pas créé.")
        return

    try:
        with open(output_csv_file, 'w', newline='', encoding='utf-8') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(['i', 'score'])
            for row in extracted_data:
                writer.writerow(row)
        print(f"Le fichier CSV '{output_csv_file}' a été créé avec succès avec {len(extracted_data)} lignes de données.")

    except Exception as e:
        print(f"Une erreur est survenue lors de l'écriture du fichier CSV : {e}")

# --- Utilisation ---
input_file = "Corpus_scores_vec_enhanced/cores_tweets_vec3.txt"
input_file_2 = "Corpus_scores_vec_enhanced/cores_tweets_vec4.txt"
output_file = "scores_output_flexible.csv"
output_file_2 = "scores_output_flexible_2.csv"
transform_txt_to_csv_flexible(input_file, output_file)
transform_txt_to_csv_flexible(input_file_2, output_file_2)