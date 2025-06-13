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

# Function to calculate BLEU score and update the same file
def calculate_bleu_scores(input_file):
    # Read the input CSV file
    with open(input_file, 'r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file, delimiter=',')
        data = list(reader)
    
    results = []
    total_score = 0.0
    
    # Iterate through each row and calculate BLEU score
    for row in data:
        ground_truth_query = preprocess_query(row["Ground Truth Query"])
        predicted_query = preprocess_query(row["Predicted Query"])
        
        # Compute BLEU score
        predictions = [predicted_query]
        references = [[ground_truth_query]]
        result = sacrebleu.compute(predictions=predictions, references=references)
        score = result["score"] / 100.0  # Normalize the BLEU score
        total_score += score
        
        # Update row with BLEU score
        row["BLEU Score"] = round(score, 4)
        results.append(row)
    
    # Calculate overall score
    overall_score = total_score / len(results) if results else 0.
    print(f"Overall BLEU Score: {overall_score:.4f}")
    
    # Write updated data back to the same CSV file
    with open(input_file, 'w', newline='', encoding='utf-8') as file:
        fieldnames = list(results[0].keys())
        writer = csv.DictWriter(file, fieldnames=fieldnames, delimiter=',')
        writer.writeheader()
        writer.writerows(results)

# Example usage
input_csv = "path_to_predictions_file.csv"  # Replace with your file path
calculate_bleu_scores(input_csv)
