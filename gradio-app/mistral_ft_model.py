from transformers import AutoTokenizer, BitsAndBytesConfig, BitsAndBytesConfig
import gradio as gr
import os
from peft import AutoPeftModelForCausalLM
import re
import torch

class MistralFtModel:
    def __init__(self):
        # Model
        self.model_id = "mistralai/Mistral-Small-24B-Instruct-2501"

        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
        )

        self.model = AutoPeftModelForCausalLM.from_pretrained("lora_adapter_24B", quantization_config=self.bnb_config, device_map={"":0})
                                             
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, token=os.environ['HF_TOKEN'])

        # Split the string on the split word "### Answer:"
        self.split_word = "[/INST]"


    def truncate_string(self, input_string):
        keyword = "###EOA"
        keyword_index = input_string.find(keyword)
        if keyword_index != -1:
            return input_string[:keyword_index]
        return input_string  # Return the original string if keyword not found
            

    def generate_answer(self, rule, key_values):
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
        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
        outputs = self.model.generate(**inputs, max_new_tokens=300)
        output_decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
        substrings = re.split(rf"{re.escape(self.split_word)}", output_decoded)
        
        predicted_query = substrings[1]
        predicted_query = self.truncate_string(predicted_query)
        return predicted_query
    
    def generate_open_answer(self, query, key_values):
        prompt =f'''<s>[SYSTEM_PROMPT]Consider the following schema information of a Neo4j graph database storing event logs:
                Node Types: 
                Event (with properties Activity, Actor, Timestamp)
                Entity (with properties EntityType, ID)
                Relationship Types: 
                Event -[:DF]-> Event
                Event -[:CORR]-> Entity
                Entity -[:REL]- Entity[/SYSTEM_PROMPT]
                    
 		[INST]For the following user query, create a corresponding Cypher query: {query}
 		The relevant key values for this query are: {key_values}
		Ensure that the query is syntactically correct, adheres to the database schema and leverages the key values effectively. [/INST]'''
        device = "cuda:0"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
        outputs = self.model.generate(**inputs, max_new_tokens=300)
        output_decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
        substrings = re.split(rf"{re.escape(self.split_word)}", output_decoded)
        
        predicted_query = substrings[1]
        predicted_query = self.truncate_string(predicted_query)
        return predicted_query
