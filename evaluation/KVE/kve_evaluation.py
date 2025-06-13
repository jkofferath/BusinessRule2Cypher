import pandas as pd
import re

def parse_key_values(key_values_str):
    """Extracts key values manually using regex."""
    if not isinstance(key_values_str, str) or not key_values_str.strip():
        return {"Activity": [], "EntityType": [], "Actor": []}

    key_values = {"Activity": [], "EntityType": [], "Actor": []}

    # Regex to extract values inside brackets, handling double quotes
    activity_match = re.search(r'Activity:\s*\[(.*?)\]', key_values_str)
    entity_match = re.search(r'EntityType:\s*\[(.*?)\]', key_values_str)
    actor_match = re.search(r'Actor:\s*\[(.*?)\]', key_values_str)

    if activity_match:
        key_values["Activity"] = [x.strip().strip('"') for x in activity_match.group(1).split(",") if x.strip()]
    if entity_match:
        key_values["EntityType"] = [x.strip().strip('"') for x in entity_match.group(1).split(",") if x.strip()]
    if actor_match:
        key_values["Actor"] = [x.strip().strip('"') for x in actor_match.group(1).split(",") if x.strip()]

    return key_values


def compute_precision_recall(gt, pred):
    """Computes precision and recall for extracted key values, handling special empty case."""
    if all(len(gt[key]) == 0 for key in gt) and all(len(pred[key]) == 0 for key in pred):
        return 1.0, 1.0  # Special case: both are empty → perfect precision and recall

    true_positives = sum(len(set(pred[key]) & set(gt[key])) for key in gt)
    predicted_count = sum(len(set(pred[key])) for key in pred)
    ground_truth_count = sum(len(set(gt[key])) for key in gt)
    
    precision = true_positives / predicted_count if predicted_count > 0 else 0.0
    recall = true_positives / ground_truth_count if ground_truth_count > 0 else 0.0
    
    return precision, recall


def evaluate_key_extraction(ground_truth_file, predictions_file, output_file):
    """Evaluates key value extraction performance and writes the results to a CSV file."""
    # Load CSV files
    gt_df = pd.read_csv(ground_truth_file, sep=',', encoding='utf-8')
    pred_df = pd.read_csv(predictions_file, sep=',', encoding='utf-8')
    
    results = []
    cumulative_true_positives = 0
    cumulative_predicted_count = 0
    cumulative_ground_truth_count = 0

    print("GT Columns:", gt_df.columns)
    print("Pred Columns:", pred_df.columns)

    for _, gt_row in gt_df.iterrows():
        nl_input = gt_row['NL input']
        gt_key_values = parse_key_values(gt_row['Key Values'])
        
        # Find corresponding prediction
        pred_row = pred_df[pred_df['NL input'] == nl_input]
        if pred_row.empty:
            pred_key_values = {"Activity": [], "EntityType": [], "Actor": []}
        else:
            pred_key_values = parse_key_values(pred_row.iloc[0]['Predicted Key Values'])
        
        precision, recall = compute_precision_recall(gt_key_values, pred_key_values)

        # Update cumulative values for overall precision and recall
        cumulative_true_positives += sum(len(set(pred_key_values[key]) & set(gt_key_values[key])) for key in gt_key_values)
        cumulative_predicted_count += sum(len(set(pred_key_values[key])) for key in pred_key_values)
        cumulative_ground_truth_count += sum(len(set(gt_key_values[key])) for key in gt_key_values)

        print("GT Parsed:", gt_key_values)
        print("Pred Parsed:", pred_key_values)

        results.append([nl_input, gt_row['Key Values'], pred_row.iloc[0]['Predicted Key Values'] if not pred_row.empty else "{}", precision, recall])
    
    # Compute overall precision and recall
    overall_precision = cumulative_true_positives / cumulative_predicted_count if cumulative_predicted_count > 0 else 0.0
    overall_recall = cumulative_true_positives / cumulative_ground_truth_count if cumulative_ground_truth_count > 0 else 0.0

    # Save results to CSV
    result_df = pd.DataFrame(results, columns=['NL input', 'GT Key Values', 'Predicted Key Values', 'Precision', 'Recall'])
    result_df.to_csv(output_file, sep=',', index=False)
    
    print(f"Results saved to {output_file}")
    print(f"Overall Precision: {overall_precision:.4f}")
    print(f"Overall Recall: {overall_recall:.4f}")


# Example usage
gt_file = 'path_to_ground_truth_file.csv' # Replace with your file path
predictions_file = 'path_to_predictions_file.csv' # Replace with your file path
results_file = 'path_to_results_file.csv' # Replace with your file path
evaluate_key_extraction(gt_file, predictions_file, results_file)
