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


def truncate_string(input_string):
    keyword = "###EOA"
    keyword_index = input_string.find(keyword)
    if keyword_index != -1:
        return input_string[:keyword_index]
    return input_string  # Return the original string if keyword not found

# Split the string on "### Answer:"
split_word = "### Answer:"

#retrieve validation set - replace the file path
validation_dataset = pd.read_csv("path_to_validatiopn_set.csv", delimiter=',')

#define data frame that will later contain the predicted query, can be used for evaluation later
df = pd.DataFrame(columns=['NL input', 'Key Values', 'Predicted Query'])

#Iterate through test_dataset, prompt the fine-tuned model to obtain the predicted query for each sample in the test dataset
for index, row in validation_dataset.iterrows():
	rule = row['NL input']
	key_values = row['Key Values']
	prompt = f'''<s>[SYSTEM_PROMPT]You are an expert in translating NL business rules into Cypher queries that check the specified property.
		Consider the following schema information of a Neo4j graph database storing event logs:
		Node Types: 
		Event (with properties Activity, Actor, Timestamp)
		Entity (with properties EntityType, ID)
		Relationship Types: 
		Event -[:DF]-> Event
		Event -[:CORR]->Entity
		Entity -[:REL]- Entity[/SYSTEM_PROMPT]
		
		[INST]I want to check the following business rule: The same actor may not perform more than 20 "Log Activity" actions.
		The relevant key values for this query are: Activity: ["log activity"], EntityType: [], Actor: []
		Create a corresponding Cypher query that returns true if the rule is satisfied, false otherwise. Ensure that the query is syntactically correct, adheres 			to the database schema and leverages the key values effectively.
		### Answer: MATCH (e:Event {{Activity: "log activity"}}) WITH e.Actor AS actor, COUNT(e) AS occurrences WHERE occurrences > 20 RETURN CASE WHEN COUNT(actor) > 0 THEN false ELSE true END AS ruleSatisfied ###EOA[/INST]
		
		[INST]I want to check the following business rule: An "End Meeting" task can only occur if a "Start Meeting" task has been performed beforehand for a given entity.
		The relevant key values for this query are: Activity: ["Start Meeting", "End Meeting"], EntityType: [], Actor: []
		Create a corresponding Cypher query that returns true if the rule is satisfied, false otherwise. Ensure that the query is syntactically correct, adheres 			to the database schema and leverages the key values effectively.
		### Answer: MATCH (e2:Event {{Activity: "End Meeting"}})-[:CORR]->(entity:Entity) OPTIONAL MATCH (e1:Event {{Activity: "Start Meeting"}})-[:CORR]->(entity:Entity) WHERE e1.timestamp < e2.timestamp WITH e2, count(e1) AS precedingStartCount RETURN CASE WHEN any(e IN collect(precedingStartCount) WHERE e = 0) THEN false ELSE true END AS ruleSatisfied ###EOA[/INST]
		
		[INST]I want to check the following business rule: The execution of activity "Create Order" and activity "Process Payment" must be separated by exactly 1 day for each entity.
		The relevant key values for this query are: Activity: ["Create Order", "Process Payment"], EntityType: [], Actor: []
		Create a corresponding Cypher query that returns true if the rule is satisfied, false otherwise. Ensure that the query is syntactically correct, adheres 			to the database schema and leverages the key values effectively.
		### Answer: MATCH (e1:Event)-[:CORR]->(entity:Entity)<-[:CORR]-(e2:Event) WHERE e1.Activity = "Create Order" AND e2.Activity = "Process Payment" AND duration.between(e1.timestamp, e2.timestamp) <> duration({{days: 1}}) RETURN CASE WHEN COUNT(e1) > 0 THEN false ELSE true END AS rule_satisfied ###EOA[/INST]
		
		[INST]I want to check the following business rule: If an article is published, each associated reviewer of that article must review an article.
		The relevant key values for this query are: Activity: ["Publish Article", "Review Article"], EntityType: ["Article", "Reviewer"], Actor: []
		Create a corresponding Cypher query that returns true if the rule is satisfied, false otherwise. Ensure that the query is syntactically correct, adheres 			to the database schema and leverages the key values effectively.
		### Answer: MATCH (article:Entity {{EntityType: "Article"}})<-[:CORR]-(publishEvent:Event {{Activity: "Publish Article"}}) MATCH (article)-[:REL]-(reviewer:Entity {{EntityType: "Reviewer"}}) WHERE NOT EXISTS {{ MATCH (reviewer)<-[:CORR]-(reviewEvent:Event {{Activity: "Review Article"}}) }} WITH COUNT(reviewer) AS reviewersWithoutReview RETURN CASE WHEN reviewersWithoutReview = 0 THEN true ELSE false END AS ruleSatisfied ###EOA[/INST]
		
		[INST]I want to check the following business rule: If an employee is hired, the onboarding process must be completed within 5 days.
		The relevant key values for this query are: Activity: ["Hiring Employee", "Onboarding Process"], EntityType: [], Actor: []
		Create a corresponding Cypher query that returns true if the rule is satisfied, false otherwise. Ensure that the query is syntactically correct, adheres 			to the database schema and leverages the key values effectively.
		### Answer: MATCH (e1:Event {{Activity: "Hiring Employee"}})-[:CORR]->(entity:Entity) OPTIONAL MATCH (e2:Event {{Activity: "Onboarding Process"}})-[:CORR]->(entity) WHERE e1.timestamp < e2.timestamp AND e2.timestamp <= e1.timestamp + duration({{days: 5}}) WITH e1, count(e2) AS succeedingCount RETURN CASE WHEN any(e IN collect(succeedingCount) WHERE e = 0) THEN false ELSE true END AS ruleSatisfied ###EOA[/INST]
		
		[INST]I want to check the following business rule: At least 3 times, when a team submits a report, it is immediately reviewed by a supervisor.
		The relevant key values for this query are: Activity: ["Submit Report", "Review Report"], EntityType: [], Actor: []
		Create a corresponding Cypher query that returns true if the rule is satisfied, false otherwise. Ensure that the query is syntactically correct, adheres 			to the database schema and leverages the key values effectively.
		### Answer: MATCH (e1:Event {{Activity: "Submit Report"}})-[:DF]->(e2:Event {{Activity:"Review Report"}}) RETURN COUNT(DISTINCT e1)>=3 AS ruleSatisfied ###EOA[/INST]
		
		[INST]I want to check the following business rule: A warehouse should not process more than ten shipments in a single day.
		The relevant key values for this query are: Activity: ["process shipment"], EntityType: ["warehouse"], Actor: []
		Create a corresponding Cypher query that returns true if the rule is satisfied, false otherwise. Ensure that the query is syntactically correct, adheres 			to the database schema and leverages the key values effectively.
		### Answer: MATCH (warehouse:Entity {{EntityType: "Warehouse"}})<-[:CORR]-(shipmentEvent:Event {{Activity: "process shipment"}}) WITH warehouse, shipmentEvent WITH warehouse, COLLECT(shipmentEvent.timestamp) AS timestamps WITH warehouse, REDUCE(ruleSatisfied = true, idx IN RANGE(0, SIZE(timestamps) - 1) | ruleSatisfied AND SIZE([t IN timestamps WHERE t >= timestamps[idx] AND t < timestamps[idx] + duration({{days: 1}})]) <= 10) AS ruleSatisfied WITH COLLECT(ruleSatisfied) AS warehouseRules RETURN ALL(rule IN warehouseRules WHERE rule = true) AS ruleHoldsForAllWarehouses ###EOA[/INST]
		
		[INST]I want to check the following business rule: A person is not allowed to perform both "Approve Loan" and "Reject Insurance Claim" for the same entity.
		The relevant key values for this query are: Activity: ["Approve Loan", "Reject Insurance Claim"], EntityType: [], Actor: []
		Create a corresponding Cypher query that returns true if the rule is satisfied, false otherwise. Ensure that the query is syntactically correct, adheres 			to the database schema and leverages the key values effectively.
		### Answer: MATCH (e1:Event {{Activity:"Approve Loan"}})-[:CORR]->(ent:Entity), (e2:Event{{Activity:"Reject Insurance Claim"}})-[:CORR]->(ent) WHERE e1.Actor = e2.Actor RETURN CASE WHEN COUNT(e1) > 0 THEN false ELSE true END AS ruleSatisfied ###EOA[/INST]
		
		[INST]I want to check the following business rule: The sojourn time for "Return Product" should be more than 1 hour on average.
		The relevant key values for this query are: Activity: ["Return Product"], EntityType: [], Actor: []
		Create a corresponding Cypher query that returns true if the rule is satisfied, false otherwise. Ensure that the query is syntactically correct, adheres 			to the database schema and leverages the key values effectively.
		### Answer: MATCH (prec:Event)-[:DF]->(e:Event {{Activity: "Return Product"}}) WITH e, MAX(prec.timestamp) AS latestPredecessor WITH e, duration.between(latestPredecessor, e.timestamp) AS sojournDuration RETURN AVG(sojournDuration.seconds) > (60 * 60) AS ruleSatisfied ###EOA[/INST]
		
		[INST]I want to check the following business rule: {rule}
		The relevant key values for this query are: {key_values}
		Create a corresponding Cypher query that returns true if the rule is satisfied, false otherwise. Ensure that the query is syntactically correct, adheres 			to the database schema and leverages the key values effectively.
		### Answer: '''
                
	device = "cuda:0"
	inputs = tokenizer(prompt, return_tensors="pt").to(device)

	outputs = model.generate(**inputs, max_new_tokens=300)
	input_length = inputs['input_ids'].shape[1]

	# Decode only the newly generated tokens
	generated_tokens = outputs[0][input_length:]
	new_tokens_decoded = tokenizer.decode(generated_tokens, skip_special_tokens=True)
	predicted_query = truncate_string(new_tokens_decoded)
	print(predicted_query)	
    
	new_entry = {
	'NL input': rule,
	'Key Values' : key_values,
	'Predicted Query': predicted_query
	}
	# Append the new entry to the DataFrame
	df = df._append(new_entry, ignore_index=True)
    
#Convert the data frame into a csv file and store it. Use it for evaluation.
df.to_csv('path_to_predictions_file.csv', sep=',', index=False) #replace the file path
