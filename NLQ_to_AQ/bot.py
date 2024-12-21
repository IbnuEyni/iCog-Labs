from dotenv import load_dotenv
import streamlit as st
import os
import google.generativeai as genai
from prompts import classify_entities_and_relationships, construct_json, extract_entities

# Load environment variables
load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

model = genai.GenerativeModel('gemini-pro')

def get_gemini_response(question, prompt):
    response = model.generate_content([prompt, question])
    return response.text

## Streamlit App

st.set_page_config(page_title="Multi-Step Annotation Query Converter")
st.header("NL-Query to Annotation Query Converter")

question = st.text_input("Input your biological query:", key="input")
submit = st.button("Convert to JSON")

if submit:
    try:
        print("Step 1: Extracted Entities and Relationships")
        entities_response = extract_entities(question)
        print(entities_response)
        
        print("Step 2: Classified Entities and Relationships")
        classified_data = classify_entities_and_relationships(entities_response)
        print(classified_data)
        
        print("Step 3: Final JSON Output")
        json_response = construct_json(classified_data)
        print(json_response)
        st.write(json_response)
        
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")