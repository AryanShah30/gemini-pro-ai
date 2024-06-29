import os
import json

import google.generativeai as genai

# getting the working directory
working_directory = os.path.dirname(os.path.abspath(__file__))

config_file_path = f"{working_directory}/config.json"
config_data = json.load(open(config_file_path))

# loading the api key
API_KEY = config_data["API_KEY"]

# configuring google.generativeai with API key
genai.configure(api_key=API_KEY)


# function to load gemini-pro-model for chatbot
def load_gemini_pro_model():
    gemini_pro_model = genai.GenerativeModel("gemini-pro")
    return gemini_pro_model


# function for image captioning
def gemini_pro_vision_response(prompt, image):
    gemini_pro_vision_model = genai.GenerativeModel("gemini-pro-vision")
    response = gemini_pro_vision_model.generate_content([prompt, image])
    result = response.text
    return result


# function to get embeddings for text
def embedding_model_response(input_text):
    embedding_model = "models/embedding-001"
    embedding = genai.embed_content(
        model=embedding_model, content=input_text, task_type="retrieval_document"
    )
    embedding_list = embedding["embedding"]
    return embedding_list


# function to get a response from gemini-pro LLM
def gemini_pro_response(user_prompt):
    gemini_pro_model = genai.GenerativeModel("gemini-pro")
    response = gemini_pro_model.generate_content(user_prompt)
    result = response.text
    return result


# function to translate language
def gemini_pro_translate(text, input_language, output_language):
    gemini_pro_model = genai.GenerativeModel("gemini-pro")
    prompt = (
        f"Translate the following '{text}' from {input_language} to {output_language}."
    )
    response = gemini_pro_model.generate_content(prompt)
    result = response.text
    return result


# function to analyse sentiments
def gemini_pro_analyse(text):
    gemini_pro_model = genai.GenerativeModel("gemini-pro")
    prompt = f"Analyse the sentiment of '{text}' and provide a concise evaluation."
    response = gemini_pro_model.generate_content(prompt)
    result = response.text
    return result


# function to summarise text
def gemini_pro_summarise(text):
    gemini_pro_model = genai.GenerativeModel("gemini-pro")
    prompt = f"Summarise the key points in '{text}'."
    response = gemini_pro_model.generate_content(prompt)
    result = response.text
    return result


# function to find NER
def gemini_pro_ner(text):
    gemini_pro_model = genai.GenerativeModel("gemini-pro")
    prompt = f"Identify and categorize named entities in '{text}', presenting them in a structured format."
    response = gemini_pro_model.generate_content(prompt)
    result = response.text
    return result


# function to check for anomaly detection
def gemini_pro_anomaly(text):
    gemini_pro_model = genai.GenerativeModel("gemini-pro")
    prompt = f"Detect any anomalies in the provided '{text}' data and describe them systematically."
    response = gemini_pro_model.generate_content(prompt)
    result = response.text
    return result


# function to beautify code
def gemini_pro_beautify(text):
    gemini_pro_model = genai.GenerativeModel("gemini-pro")
    prompt = f"Beautify the code given as {text} in a nice manner. Do not make any changes to the code."
    response = gemini_pro_model.generate_content(prompt)
    result = response.text
    return result


# function to correct code
def gemini_pro_correct(text):
    gemini_pro_model = genai.GenerativeModel("gemini-pro")
    prompt = f"Identify errors in the code snippet '{text}' and suggest corrections."
    response = gemini_pro_model.generate_content(prompt)
    result = response.text
    return result

