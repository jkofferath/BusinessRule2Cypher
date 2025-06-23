import csv
import threading
import ast
from neo4j import GraphDatabase

# Neo4j connection details (replace with your credentials)
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "12341234"
TIMEOUT = 120  # Timeout in seconds

class Neo4jExecutor:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
    
    def close(self):
        self.driver.close()
    
    def execute_query(self, query):
        """Executes a Cypher query with a timeout."""
        print(f"Try to execute query {query}")
        result_container = []

        def run_query():
            try:
                with self.driver.session() as session:
                    result = session.run(query)
                    result_container.append(str([record.data() for record in result]))
            except Exception as e:
                result_container.append(f"Error: {str(e)}")
        
        thread = threading.Thread(target=run_query)
        thread.start()
        thread.join(timeout=TIMEOUT)
        
        if not result_container:
            return "Error: Query timeout"
        return result_container[0]

# Postprocess query result
import ast

def process_result(result):
    """Processes the query execution result."""
    
    try:
        # Only evaluate if it looks like a list
        if isinstance(result, str) and result.strip().startswith('[') and result.strip().endswith(']'):
            evaluated_result = ast.literal_eval(result)
            if isinstance(evaluated_result, list):
                if len(evaluated_result) > 1:  # List of multiple boolean values
                    return "Other"
        
        # Other string-based checks
        if "True" in result:
            return "True"
        elif "False" in result:
            return "False"
        elif "None" in result:
            return "none"
        elif "timeout" in result:
            return "timeout"
        else:
            return "Error"

    except (ValueError, SyntaxError):
        # If literal_eval fails, check for known string cases
        if "None" in result:
            return "none"
        elif "timeout" in result:
            return "timeout"
        else:
            return "Error"


# Load CSV data
def load_csv(file_path, query_column):
    """Loads CSV data and extracts relevant columns."""
    data = []
    with open(file_path, newline='', encoding='utf-8-sig') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',')
        headers = reader.fieldnames
        if headers:
            headers = [h.strip() for h in headers]
            print(f"Detected CSV Headers: {headers}")
        
        for row in reader:
            try:
                data.append({
                    "NL input": row["NL input"].strip(),
                    "query": row[query_column].strip(),
                    "key_values": row.get("Key Values", "").strip()
                })
            except KeyError as e:
                print(f"Error: Missing column {str(e)} in file {file_path}. Available columns: {headers}")
                raise
    return data

def parse_key_values(kv_string):
    """Parses the key value string into a dictionary."""
    if not kv_string:
        return {}
    
    result = {}
    current_key = None
    current_value = ""
    in_brackets = False

    i = 0
    while i < len(kv_string):
        char = kv_string[i]

        if char == ":" and not in_brackets:
            current_key = kv_string[:i].strip()
            kv_string = kv_string[i + 1:].lstrip()
            i = 0
            continue
        elif char == "[":
            in_brackets = True
            current_value = ""
        elif char == "]":
            in_brackets = False
            try:
                result[current_key] = ast.literal_eval("[" + current_value + "]")
            except Exception as e:
                print(f"Error parsing list for key '{current_key}': {e}")
                result[current_key] = []
            kv_string = kv_string[i + 1:].lstrip(", ").lstrip()
            i = 0
            current_key = None
            continue
        elif in_brackets:
            current_value += char

        i += 1

    return result

# Check if predicted query contains all key values
def contains_all_key_values(pred_query, key_values_dict):
    """Checks if all values from key_values_dict appear in the predicted query."""
    for key, values in key_values_dict.items():
        for value in values:
            if value not in pred_query:
                return False
    return True

# Compare results,  write output and calculate metrics
def compare_and_save_results(gt_file, pred_file, output_file):
    gt_data = load_csv(gt_file, "Cypher Query")
    pred_data = load_csv(pred_file, "Predicted Query")
    
    if len(gt_data) != len(pred_data):
        raise ValueError("Mismatch in row count between ground truth and predicted files.")
    
    executor = Neo4jExecutor(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

    exact_match_count = 0
    syntax_error_count = 0
    all_key_values_used_count = 0
    total_count = len(gt_data)

    with open(output_file, mode='w', newline='', encoding='utf-8-sig') as csvfile:
        fieldnames = [
            "NL input", "Ground Truth Query", "GT Result",
            "Predicted Query", "Predicted Result", "Exact Match",
            "Key Values Match"
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=',')
        writer.writeheader()
        
        for gt, pred in zip(gt_data, pred_data):
            gt_result = executor.execute_query(gt["query"])
            pred_result = executor.execute_query(pred["query"])
            
            processed_gt_result = process_result(gt_result)
            processed_pred_result = process_result(pred_result)
            if(processed_gt_result != "timeout" and processed_pred_result != "timeout"):
                exact_match = processed_gt_result == processed_pred_result
            else:
                exact_match = False

            key_values_dict = parse_key_values(gt["key_values"])
            key_values_match = contains_all_key_values(pred["query"], key_values_dict)
            if key_values_match:
                all_key_values_used_count += 1

            if exact_match:
                exact_match_count += 1
            if processed_pred_result == "Error":
                syntax_error_count += 1

            writer.writerow({
                "NL input": gt["NL input"],
                "Ground Truth Query": gt["query"],
                "GT Result": processed_gt_result,
                "Predicted Query": pred["query"],
                "Predicted Result": processed_pred_result,
                "Exact Match": exact_match,
                "Key Values Match": key_values_match
            })
    
    executor.close()

    exact_match_percentage = (exact_match_count / total_count) * 100
    syntax_error_rate = (syntax_error_count / total_count) * 100
    all_key_values_used_rate = (all_key_values_used_count /total_count) * 100

    print(f"\nEvaluation Summary:")
    print(f"Total queries evaluated: {total_count}")
    print(f"Exact matches of query execution: {exact_match_count} / {total_count} ({exact_match_percentage:.2f}%)")
    print(f"Predicted syntax errors: {syntax_error_count} / {total_count} ({syntax_error_rate:.2f}%)")
    print(f"Ratio of samples which use all relevant key values: {all_key_values_used_count} / {total_count} ({all_key_values_used_rate:.2f}%)")
    print(f"\nResults saved to {output_file}")

# Usage: As input, we require a ground truth file which follows the same schema as the validation sets, and a predictions file. This should look like the example_predictions_file.csv file.
if __name__ == "__main__":
    compare_and_save_results(
        "path_to_ground_truth_file.csv", # Replace with your input path
        "path_to_predictions_file.csv", # Replace with your input path
        "path_to_results_file.csv" # Replace with your file path
    )
