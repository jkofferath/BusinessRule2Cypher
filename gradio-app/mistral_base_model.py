from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, BitsAndBytesConfig
import torch
import os

class MistralBaseModel:
    def __init__(self):
        self.model_id = "mistralai/Mistral-Small-24B-Instruct-2501"

        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
        )

        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, quantization_config=self.bnb_config,
                                             device_map={"":1},
                                             token=os.environ['HF_TOKEN'])
                                             
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, token=os.environ['HF_TOKEN'])

        # Split the string on the split word "### Answer:"
        self.split_word = "### Answer:"


    def truncate_string(self, input_string):
        keyword = "###EOA"
        keyword_index = input_string.find(keyword)
        if keyword_index != -1:
            return input_string[:keyword_index]
        return input_string  # Return the original string if keyword not found
    
    def explain_query(self, cypher_query):
        prompt =f"""
        You are an expert in Neo4j and Cypher query language.

        **Task:**  
        Analyze the following Cypher query and explain in simple terms what it does:  

        ```cypher
        {cypher_query}
        ```

        **Instructions:**  
        - Provide a **concise**, natural language explanation.  
        - Clearly describe what data it retrieves, modifies, or processes.  
        - If it includes filters (e.g., `WHERE`), explain their effect.  
        - If it involves relationships, describe how nodes are connected.  
        - **Do NOT include code** in your response—just the explanation.

        ### Answer:
        """
        device = "cuda:1"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
        outputs = self.model.generate(**inputs, max_new_tokens=600)
        print(self.tokenizer.decode(outputs[0], skip_special_tokens=True))
        input_length = inputs['input_ids'].shape[1]
        
        # Decode only the newly generated tokens
        generated_tokens = outputs[0][input_length:]
        new_tokens_decoded = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return new_tokens_decoded

    def correct_error(self, user_request, error_message):
        prompt =f"""
        You are an expert in Neo4j and Cypher query language.

        **Task:**  
        Consider the following user request in NL together with a Neo4j error message upon the execution of a Cypher query. Provide a corrected version of the Cypher query and a very short explanation of the error cause.

        User Request: {user_request}
        Error Message: {error_message}
        

        **Instructions:**  
        - Start with the corrected query.
        - Explain in 1-2 sentences what the problem was. Explain using non-technical terms.

        ### Answer:
        """
        device = "cuda:1"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
        outputs = self.model.generate(**inputs, max_new_tokens=600)
        print(self.tokenizer.decode(outputs[0], skip_special_tokens=True))
        input_length = inputs['input_ids'].shape[1]
        
        # Decode only the newly generated tokens
        generated_tokens = outputs[0][input_length:]
        new_tokens_decoded = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return new_tokens_decoded

    def prettify_result(self, result):
        prompt= f"""You are an expert in prettifying outputs from a Cypher Query Execution. Here are some examples: 
        1. Cypher Query Output:'(ruleSatisfied: 'true')
            Answer: The rule is satisfied. ###EOA
        2. Cypher Query Output:'(ruleSatisfied: 'false')
            Answer: The rule is not satisfied. ###EOA
        
       	In case of an error, mention that there was an error, display the error message (the complete one, with the original query), and shortly explain why the error occurred using non-technical terms.
       	
       	Now consider this output of a Cypher query execution:
        Cypher Query Output:{result} 
            Answer:"""
        device = "cuda:1"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
        outputs = self.model.generate(**inputs, max_new_tokens=200)
        print(self.tokenizer.decode(outputs[0], skip_special_tokens=True))
        input_length = inputs['input_ids'].shape[1]
        
        # Decode only the newly generated tokens
        generated_tokens = outputs[0][input_length:]
        new_tokens_decoded = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        key_values = self.truncate_string(new_tokens_decoded)
        return key_values
        

    def generate_answer_kve(self, available_key_values, rule):
        prompt=prompt=f""" You are an expert in extracting key values from natural language (NL) inputs.

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
        
        device = "cuda:1"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
        outputs = self.model.generate(**inputs, max_new_tokens=100)
        print(self.tokenizer.decode(outputs[0], skip_special_tokens=True))
        input_length = inputs['input_ids'].shape[1]
        
        # Decode only the newly generated tokens
        generated_tokens = outputs[0][input_length:]
        new_tokens_decoded = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        key_values = self.truncate_string(new_tokens_decoded)
        return key_values
