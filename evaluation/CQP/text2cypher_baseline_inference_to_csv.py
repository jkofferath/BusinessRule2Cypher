from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, BitsAndBytesConfig
import torch
import pandas as pd


baseline_model_name = "neo4j/text2cypher-gemma-2-9b-it-finetuned-2024v1"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
baseline_tokenizer = AutoTokenizer.from_pretrained(baseline_model_name)
baseline_model = AutoModelForCausalLM.from_pretrained(
baseline_model_name,
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16,
    attn_implementation="eager",
    low_cpu_mem_usage=True,
)

# Any other parameters
model_generate_parameters = {
    "top_p": 0.9,
    "temperature": 0.2,
    "max_new_tokens": 512,
    "do_sample": True,
    "pad_token_id": baseline_tokenizer.eos_token_id,
}

instruction = (
    "Generate Cypher statement to query a graph database. "
    "Use only the provided relationship types and properties in the schema. \n"
    "Schema: {schema} \n Question: {question}  \n Cypher output: "
)

def prepare_chat_prompt(question, schema) -> list[dict]:
	chat = [
	    {
		"role": "user",
		"content": instruction.format(
		    schema=schema, question=question
		),
	    }
	]
	return chat

def _postprocess_output_cypher(output_cypher: str) -> str:
	# Remove any explanation. E.g.  MATCH...\n\n**Explanation:**\n\n -> MATCH...
	# Remove cypher indicator. E.g.```cypher\nMATCH...```` --> MATCH...
	# Note: Possible to have both:
	#   E.g. ```cypher\nMATCH...````\n\n**Explanation:**\n\n --> MATCH...
	partition_by = "**Explanation:**"
	output_cypher, _, _ = output_cypher.partition(partition_by)
	output_cypher = output_cypher.strip("`\n")
	output_cypher = output_cypher.lstrip("cypher\n")
	output_cypher = output_cypher.strip("`\n ")
	return output_cypher


def generate_answer(rule, key_values):
	question = f"{rule} Return true if this holds, false otherwise. Key values: {key_values}"
	schema = """Node Types: 
		Event (with properties Activity, Actor, timestamp)
	    Entity (with properties EntityType, ID)
	    Relationship Types: 
	    Event -[:DF]-> Event 
	    Event -[:CORR]->Entity
	    Entity -[:REL]- Entity
	    """
	new_message = prepare_chat_prompt(question=question, schema=schema)
	prompt = baseline_tokenizer.apply_chat_template(new_message, add_generation_prompt=True, tokenize=False)
	inputs = baseline_tokenizer(prompt, return_tensors="pt", padding=True)

	inputs.to(baseline_model.device)
	baseline_model.eval()

	with torch.no_grad():
		tokens = baseline_model.generate(**inputs, **model_generate_parameters)
		tokens = tokens[:, inputs.input_ids.shape[1] :]
		raw_outputs = baseline_tokenizer.batch_decode(tokens, skip_special_tokens=True)
		outputs = [_postprocess_output_cypher(output) for output in raw_outputs]
		output = outputs[0]
	
	return(output)
	
#retrieve validation set - replace the file path
validation_dataset = pd.read_csv("path_to_validation_set.csv", delimiter=',')

#define data frame that will later contain the predicted query, can be used for evaluation later
df = pd.DataFrame(columns=['NL input', 'Key Values', 'Predicted Query'])

#Iterate through test_dataset, prompt the fine-tuned model to obtain the predicted query for each sample in the test dataset
for index, row in validation_dataset.iterrows():
	rule = row['NL input']
	key_values = row['Key Values']
	predicted_query = generate_answer(rule, key_values)
	print(predicted_query)	
    
	new_entry = {
	'NL input': rule,
	'Key Values' : key_values,
	'Predicted Query': predicted_query
	}
	# Append the new entry to the DataFrame
	df = df._append(new_entry, ignore_index=True)
    
#Convert the data frame into a csv file and store it. Use it for evaluation.
df.to_csv('path_to_predictions_file.csv', sep=',', index=False)
