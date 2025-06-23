from transformers import AutoTokenizer, BitsAndBytesConfig, BitsAndBytesConfig
import os
from peft import AutoPeftModelForCausalLM
import re
import torch
import pandas as pd

os.environ["HF_TOKEN"] = "YOUR_HF_TOKEN"

#specify the model id of the base model, in this case Mistral's 7B variant
model_id = "mistralai/Mistral-7B-Instruct-v0.3"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
)

#specify the path to the folder that contains the LoRA weights
model = AutoPeftModelForCausalLM.from_pretrained("path_to_lora_adapter", quantization_config=bnb_config)
                                     
tokenizer = AutoTokenizer.from_pretrained(model_id, token=os.environ['HF_TOKEN'])

def truncate_string(input_string):
    keyword = "###EOA"
    keyword_index = input_string.find(keyword)
    if keyword_index != -1:
        return input_string[:keyword_index]
    return input_string  # Return the original string if keyword not found

# Split the string on the last word of the input prompt
split_word = "effectively."

#retrieve validation set - replace the file path
validation_dataset = pd.read_csv("path_to_validation_set.csv", delimiter=',')

#define data frame that will later contain the predicted query, can be used for evaluation later
df = pd.DataFrame(columns=['NL input', 'Key Values', 'Predicted Query'])

#Iterate through test_dataset, prompt the fine-tuned model to obtain the predicted query instruction for each sentence in the test dataset
for index, row in validation_dataset.iterrows():
    rule = row['NL input']
    key_values = row['Key Values']
    prompt =f'''<s>[SYSTEM_PROMPT]Consider the following schema information of a Neo4j graph database storing event logs:
                Node Types: 
                Event (with properties Activity, Actor, Timestamp)
                Entity (with properties EntityType, ID)
                Relationship Types: 
                Event -[:DF]-> Event
                Event -[:CORR]-> Entity
                Entity -[:REL]- Entity[/SYSTEM_PROMPT]
                [INST]I want to check the following business rule: {rule}
                The relevant key values for this query are: {key_values}
                Create a corresponding Cypher query that returns true if the rule is satisfied, false otherwise. Ensure that the query is syntactically correct, adheres to the database schema and leverages the key values effectively.[/INST]'''
    device = "cuda:0"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    outputs = model.generate(**inputs, max_new_tokens=300)
    output_decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    substrings = re.split(rf"{re.escape(split_word)}", output_decoded)
    
    predicted_query = substrings[1]
    predicted_query = truncate_string(predicted_query)
    print(predicted_query)
    
    new_entry = {
        'NL input': rule,
        'Key Values' : key_values,
        'Predicted Query': predicted_query
    }
    # Append the new entry to the DataFrame
    df = df._append(new_entry, ignore_index=True)
    
#Convert the data frame into a csv file and store it. Use it for evaluation.
df.to_csv('path_to_your_predictions_file.csv', sep=',', index=False) #replace the file path
