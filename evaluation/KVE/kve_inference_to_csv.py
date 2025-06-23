from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, BitsAndBytesConfig
import os
import re
import torch
import pandas as pd


os.environ["HF_TOKEN"] = "YOUR_HF_TOKEN"

model_id = "mistralai/Mistral-Small-24B-Instruct-2501"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config,
                                             device_map={"":0},
                                             token=os.environ['HF_TOKEN'])
                                     
tokenizer = AutoTokenizer.from_pretrained(model_id, token=os.environ['HF_TOKEN'])

#specify available key values here
available_key_values = """
Activity: [...]
EntityType: [...]
Actor: [...]
"""

def truncate_string(input_string):
    keyword = "###EOA"
    keyword_index = input_string.find(keyword)
    if keyword_index != -1:
        return input_string[:keyword_index]
    return input_string  # Return the original string if keyword not found


#retrieve validation set - replace the file path
validation_dataset = pd.read_csv("path_to_validation_set.csv", delimiter=',')

#define data frame that will later contain the predicted query, can be used for evaluation later
df = pd.DataFrame(columns=['NL input', 'Predicted Key Values'])

#Iterate through test_dataset, prompt the fine-tuned model to obtain the predicted query for each sample in the test dataset
for index, row in validation_dataset.iterrows():
	rule = row['NL input']
	prompt=f""" You are an expert in extracting key values from natural language (NL) inputs.

		### Task Description:
		Extract key values from the given business rule using only the available lists of:
		- Activity
		- EntityType
		- Actor

		### Example 1:
		Available Key Values:
		    Activity: ["Confirm Order", "Create Order", "place SO", "Create Invoice",
			    "Receive SO", "Update SO", "Unpack", "Update Invoice",
			    "pack shipment", "Ship", "Receive Payment", "Clear Invoice"]
		    EntityType: ["Order", "SupplierOrder", "Item", "Invoice", "Payment"]
		    Actor: ["R1", "R3", "R2", "R4", "R5"]

		#### NL Input 1.1 (Business Rule):
		"At least 2 different people must handle 'Place SO' events."
		Extracted Key Values:
		Activity: ["place SO"], EntityType: [], Actor: []
		###EOA

		#### NL Input 1.2 (Business Rule):
		"The creation of an order is mandatory for R1."
		Extracted Key Values:
		Activity: ["Create Order"], EntityType: [], Actor: ["R1"]
		###EOA

		#### NL Input 1.3 (Business Rule):
		"A person may not work on more than 3 orders."
		Extracted Key Values:
		Activity: [], EntityType: ["Order"], Actor: []
		###EOA

		#### NL Input 1.4 (Business Rule):
		"If an SO is updated, a shipment may not be packed afterward." 
		Extracted Key Values:
		Activity: ["Update SO", "pack shipment"], EntityType: [], Actor: []
		###EOA

		#### NL Input 1.5 (Business Rule):
		"No more than 7 orders may be generated."  
		Extracted Key Values:
		Activity: ["Create Order"], EntityType: [], Actor: []
		###EOA

		#### NL Input 1.6 (Business Rule):
		"Returning an item must happen at most 100 times per week."
		Extracted Key Values: 
		Activity: [], EntityType: [], Actor: []
		###EOA

		#### NL Input 1.7 (Business Rule):
		"For every item, a payment must be received."  
		Extracted Key Values:
		Activity: ["Receive Payment"], EntityType: ["Item"], Actor: []
		###EOA

		---

		### Example 2:
		Available Key Values:
		    Activity: ["Act1", "Act2", "Act3", "act4", "Act5", "Act6"]
		    EntityType: ["et1", "ET2", "et3", "ET4", "ET5"]
		    Actor: ["X", "Y", "Z", "A", "B"]

		#### NL Input 2.1 (Business Rule):
		"It is required that an event performed by Act5 must be directly succeeded
		    by an event performed by Act1 at most 2 times." 
		Extracted Key Values:
		Activity: ["Act5", "Act1"], EntityType: [], Actor: []
		###EOA

		#### NL Input 2.2 (Business Rule):
		"For every ET4, if Act2 occurs, Act6 must also be performed."
		Extracted Key Values: 
		Activity: ["Act2", "Act6"], EntityType: ["ET4"], Actor: []  
		###EOA

		#### NL Input 2.3 (Business Rule):
		"An ET3 is correlated to at most three ET4 objects." 
		Extracted Key Values:
		Activity: [], EntityType: ["et3", "ET4"], Actor: []  
		###EOA

		---

		### Example 3:
		Available Key Values:
		    {available_key_values}

		#### NL Input 3.1 (Business Rule):
		"{rule}"
		Extracted Key Values:  
	"""
	
	device = "cuda:0"
	inputs = tokenizer(prompt, return_tensors="pt").to(device)
	outputs = model.generate(**inputs, max_new_tokens=100)
	input_length = inputs['input_ids'].shape[1]

	# Decode only the newly generated tokens
	generated_tokens = outputs[0][input_length:]
	new_tokens_decoded = tokenizer.decode(generated_tokens, skip_special_tokens=True)
	key_values = truncate_string(new_tokens_decoded)
	print(key_values)

	new_entry = {
	'NL input': rule,
	'Predicted Key Values': key_values
	}
	# Append the new entry to the DataFrame
	df = df._append(new_entry, ignore_index=True)
    
#Convert the data frame into a csv file and store it. Use it for evaluation.
df.to_csv('path_to_predictions_file.csv', sep=',', index=False) # Replace with your file path
