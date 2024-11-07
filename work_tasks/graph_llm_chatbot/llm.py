import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
import streamlit as st
from graph import graph

load_dotenv()

# Create the LLM
llm =  ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        temperature=0, 
        max_tokens=None, 
        timeout=None,
        google_api_key = os.getenv('GEMINI_API_KEY')
    )

