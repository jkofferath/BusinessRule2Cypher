import csv
import evaluate

sacrebleu = evaluate.load("sacrebleu")

def preprocess_query(query):
    """
    Preprocess the query by replacing double quotes and
    normalizing whitespace.
    """
    query = query.replace("''", "'")  # Replace '' with '
    query = query.strip()  # Remove leading and trailing spaces
    query = " ".join(query.split())  # Normalize internal whitespace
    return query

def load_csv_as_dict(filepath, key_column):
    """
    Load a CSV file and return a dictionary keyed by `key_column`.
    """
    data = {}
    with open(filepath, 'r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            data[row[key_column]] = row
    return data

def calculate_bleu_scores(ground_truth_file, predictions_file, output_file):
    # Load ground truth and prediction files
    ground_truth_data = load_csv_as_dict(ground_truth_file, key_column="NL input")
    predictions_data = load_csv_as_dict(predictions_file, key_column="NL input")

    results = []
    total_score = 0.0
    match_count = 0

    for nl_input, pred_row in predictions_data.items():
        if nl_input not in ground_truth_data:
            print(f"Warning: No ground truth for NL input: {nl_input}")
            continue
        
        gt_row = ground_truth_data[nl_input]
        ground_truth_query = preprocess_query(gt_row["Cypher Query"])
        predicted_query = preprocess_query(pred_row["Predicted Query"])

        predictions = [predicted_query]
        references = [[ground_truth_query]]
        result = sacrebleu.compute(predictions=predictions, references=references)
        score = result["score"] / 100.0  # Normalize the BLEU score
        total_score += score
        match_count += 1

        # Build output row
        result_row = {
            "NL input": nl_input,
            "Cypher Query": ground_truth_query,
            "Predicted Query": predicted_query,
            "BLEU Score": round(score, 4)
        }
        results.append(result_row)

    overall_score = total_score / match_count if match_count else 0.0
    print(f"Overall BLEU Score: {overall_score:.4f}")

    # Write output CSV
    with open(output_file, 'w', newline='', encoding='utf-8-sig') as file:
        fieldnames = ["NL input", "Cypher Query", "Predicted Query", "BLEU Score"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

# Usage: As input, we require a ground truth file which follows the same schema as the validation sets, and a predictions file. This should look like the example_predictions_file.csv file.
ground_truth_csv = "path_to_ground_truth_file.csv"        # Replace with actual path
predictions_csv = "path_to_predictions_file.csv"          # Replace with actual path
output_csv = "path_to_results_file.csv"        # Output file with BLEU scores

calculate_bleu_scores(ground_truth_csv, predictions_csv, output_csv)
