
import streamlit as st
import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter
import numpy as np
from pypdf import PdfReader
from chroma import Chroma
import PyPDF2
from tqdm import tqdm
from sentence_transformers import CrossEncoder
import umap
import matplotlib.pyplot as plt
import google.generativeai as genai
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from doc_loader import PDFLoader
from embedding import GeminiEmbeddingFunction


genai.configure(api_key=st.secrets['GEMINI_API_KEY'])

import re
def split_text(text:str):
    split_text = re.split('\n',text)
    return [i for i in split_text if i!=""]

# Function to create an index from a PDF
def create_index_pdf(file_path):
    pdf_loader = PDFLoader(file_path=file_path)
    text = pdf_loader.load().content

    # Split/chunk the text
    chunked_text = split_text(text)

    # Create index
    chroma_instance = Chroma(embedding_function=GeminiEmbeddingFunction())
    collection_name = chroma_instance.add(chunked_text)
    return collection_name

# Read PDF and extract texts
def _read_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    
    pdf_texts = [p.extract_text().strip() for p in reader.pages]

    # Filter the empty strings
    pdf_texts = " ".join(pdf_texts)
    return pdf_texts


# Split texts into chunks using both character and token-based strategies
def _chunk_texts(texts):
    character_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size=1000,
        chunk_overlap=0
    )
    character_split_texts = character_splitter.split_text('\n\n'.join(texts))

    token_splitter = SentenceTransformersTokenTextSplitter(chunk_overlap=0, tokens_per_chunk=256)
    token_split_texts = []
    
    for text in character_split_texts:
        token_split_texts += token_splitter.split_text(text)

    return token_split_texts


# Load Chroma collection with chunked texts and embedding function
def load_chroma(uploaded_file, collection_name, embedding_function):
    texts = _read_pdf(uploaded_file)
    chunks = _chunk_texts(texts)

    chroma_cliet = chromadb.Client()
    chroma_collection = chroma_cliet.create_collection(name=collection_name, embedding_function=embedding_function)

    ids = [str(i) for i in range(len(chunks))]
    chroma_collection.add(ids=ids, documents=chunks)

    return chroma_collection


# Word wrap utility function for better text display
def word_wrap(string, n_chars=72):
    if len(string) < n_chars:
        return string
    else:
        return string[:n_chars].rsplit(' ', 1)[0] + '\n' + word_wrap(string[len(string[:n_chars].rsplit(' ', 1)[0])+1:], n_chars)


# Project embeddings using UMAP dimensionality reduction
def project_embeddings(embeddings, umap_transform):
    umap_embeddings = np.empty((len(embeddings), 2))
    for i, embedding in enumerate(tqdm(embeddings)): 
        umap_embeddings[i] = umap_transform.transform([embedding])
    return umap_embeddings

# Configure the embedding function for SentenceTransformer
embedding_function = SentenceTransformerEmbeddingFunction()


# Query Expansion using Generative AI
def expand_query_with_answer(query):
    prompt = f"""You are a helpful expert financial research assistant. Provide an example answer to the given question, \
    that might be found in a document like an annual report of Microsoft. Keep it very simple and generic.
    Question: {query}"""
    model = genai.GenerativeModel('gemini-pro')
    answer = model.generate_content(prompt)
    return f"{query} {answer.text}"


def expand_query_with_answer(query):
    prompt = f"""You are a helpful expert financial research assistant. Provide an example answer to the given question, \
    that might be found in a document like an annual report of Microsoft. Keep it very simple and generic.
    Question: {query}"""
    model = genai.GenerativeModel('gemini-pro')
    answer = model.generate_content(prompt)
    return f"{query} {answer.text}"

def augment_multiple_query(query):
            prompt = f"""Suggest up to five additional related questions to help them find the information they need for the provided question. 
                        Suggest a variety of short questions related to the original query.
                        Query: {query}"""
            model = genai.GenerativeModel('gemini-pro')
            answer = model.generate_content(prompt)    
            return answer.text.split("\n")

# Re-ranking documents using CrossEncoder
def rank_documents(cross_encoder: CrossEncoder, query: str, retrieved_documents: list):
    pairs = [[query, doc] for doc in retrieved_documents]
    scores = cross_encoder.predict(pairs)
    ranks = np.argsort(scores)[::-1]  # Sort in descending order
    ranked_docs = {rank: doc for rank, doc in zip(ranks, retrieved_documents)}
    return ranked_docs