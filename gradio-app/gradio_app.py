from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, BitsAndBytesConfig
import gradio as gr
import os
from neo4j_connector import Neo4jConnector
from mistral_base_model import MistralBaseModel
from mistral_ft_model import MistralFtModel

os.environ["HF_TOKEN"] = "YOUR_HF_TOKEN" # fill in your HF token
identifiers= ""
mistral_base_model = MistralBaseModel()
mistral_ft_model = MistralFtModel()


def get_query_result(query:str):
	connector = Neo4jConnector("bolt://localhost:7687", "neo4j", "12341234") # fill in connection details
	results = connector.execute_query(query)
	connector.close()
	return results

def fetch_db_identifiers() -> str:
	global identifiers
	identifiers = ""
	activity_list = get_query_result("MATCH (e:Event) RETURN DISTINCT e.Activity AS Activity")
	activities = [activity['Activity'] for activity in activity_list]
	activity_labels = f"Activity: {activities}"
	identifiers = identifiers + activity_labels + "\n"
	
	entity_type_list = get_query_result("MATCH (e:Entity) RETURN DISTINCT e.EntityType AS EntityType")
	entity_types = [ent['EntityType'] for ent in entity_type_list]
	entity_type_labels = f"EntityType: {entity_types}"
	identifiers = identifiers + entity_type_labels + "\n"
	
	actor_list = get_query_result("MATCH (e:Event) RETURN DISTINCT e.Actor AS Actor")
	if(actor_list[0]["Actor"] is None):
		identifiers = identifiers + "Actor: []" + "\n"
	else:
		actors = [actor['Actor'] for actor in actor_list]
		actor_labels = f"Actor: {actors}"
		identifiers = identifiers + actor_labels 
	global_status = f"Connected to Neo4j instance. \nAvaliable DB identifiers: \n{identifiers}"
	return global_status


def get_relevant_key_values(available_key_values, rule):
    outputs = mistral_base_model.generate_answer_kve(available_key_values, rule)
    return outputs

def get_query_from_ft_model(rule, key_values):
	outputs= mistral_ft_model.generate_answer(rule, key_values)
	return outputs

def get_open_query_from_ft_model(query, key_values):
	outputs= mistral_ft_model.generate_open_answer(query, key_values)
	return outputs

def prettify_result(result) -> str:
	outputs = mistral_base_model.prettify_result(result)
	return outputs
	
def correct_error(user_request, error_message):
	outputs = mistral_base_model.correct_error(user_request, error_message)
	return outputs

def get_custom_query_and_explain(query, key_values):
    # Step 1: Generate the Cypher query
    generated_query = get_open_query_from_ft_model(query, key_values)
    
    # Step 2: Process the generated query
    processed_result = mistral_base_model.explain_query(generated_query)
    
    return generated_query, processed_result

def direct_check(rule):
    status_updates = []
    
    # Step 1: Extracting relevant key values
    status_updates.append("1. Extracting relevant key values...")
    yield "\n".join(status_updates)
    
    relevant_key_values = get_relevant_key_values(identifiers, rule)
    status_updates.append(f"2. The relevant key values are: \n {relevant_key_values}")
    yield "\n".join(status_updates)
    
    # Step 2: Generating Cypher Query
    status_updates.append("3. Generating the Cypher query...")
    yield "\n".join(status_updates)
    
    predicted_query = get_query_from_ft_model(rule, relevant_key_values)
    print(predicted_query)
    status_updates.append(f"4. The resulting Cypher query is: \n {predicted_query}")
    yield "\n".join(status_updates)
    
    # Step 3: Executing Query
    status_updates.append("\n5. Executing the query... \n ")
    yield "\n".join(status_updates)
    
    result = get_query_result(predicted_query)
    
    # Step 4: Prettify result
    prettified_result = prettify_result(result)
    status_updates.append(f"Final result: {prettified_result}")
    
    yield "\n".join(status_updates)  # Final result update


with gr.Blocks() as app:

	with gr.Row():
		global_status = gr.Textbox(label="Global Status", value="Not connected with Neo4j.", interactive=False, lines=3, scale=3)
		connect_button = gr.Button("Connect to DB", scale=1)
		connect_button.click(fn=fetch_db_identifiers, inputs=[], outputs=[global_status])
		
	with gr.Row():
		with gr.Tab("Rule Check"):
			gr.Markdown("### Business Rule Compliance Check")
			rule = gr.Textbox(label="Business Rule in NL")
			text_output = gr.Textbox(label="Result", lines=8)
			text_generate_btn = gr.Button("Show Result")
			text_generate_btn.click(direct_check, inputs=[rule], outputs=text_output, show_progress=True)

	with gr.Row():
	    
		with gr.Tab("Key Value Extraction"):
			gr.Markdown("### Extract Key Values")
			db_key_values= gr.Textbox(label="DB identifiers")
			rule = gr.Textbox(label="Business Rule")
			text_output = gr.Textbox(label="Relevant Key Values", lines=5)
			text_generate_btn = gr.Button("Extract")
			text_generate_btn.click(get_relevant_key_values, inputs=[db_key_values, rule], outputs=text_output)
		
		with gr.Tab("Cypher Query Prediction"):
			gr.Markdown("### Generate Cypher Query")
			rule = gr.Textbox(label="Business Rule")
			key_values = gr.Textbox(label="Key Values")
			answer_output = gr.Textbox(label="Generated Query", lines=5)
			generate_btn = gr.Button("Generate")
			generate_btn.click(get_query_from_ft_model, inputs=[rule, key_values], outputs=answer_output)
		
		with gr.Tab("Custom Query"):
			gr.Markdown("### Custom Query Generation")
			query = gr.Textbox(label="Your Query in NL")
			key_values = gr.Textbox(label="Key Values")
			answer_output = gr.Textbox(label="Generated Query", lines=5) 
			processed_output = gr.Textbox(label="Explanation", lines=5) 
			text_generate_btn = gr.Button("Generate")
			text_generate_btn.click(get_custom_query_and_explain, inputs=[query, key_values], outputs=[answer_output, processed_output])
			
		with gr.Tab("Error Correction"):
			gr.Markdown("### Error Correction")
			user_request = gr.Textbox(label="User Request")
			error_message = gr.Textbox(label="Error Message")
			answer_output = gr.Textbox(label="LLM Response", lines=5)
			generate_btn = gr.Button("Generate Answer")
			generate_btn.click(correct_error, inputs=[user_request, error_message], outputs=answer_output)

# Launch the app with public sharing
app.launch(share=True)

