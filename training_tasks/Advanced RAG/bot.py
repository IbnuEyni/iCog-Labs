import tempfile
import streamlit as st
import umap
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from utils import *
from utils import _read_pdf
from sentence_transformers import CrossEncoder

# Streamlit UI
st.title("PDF Document Query and Ranking")

# File upload
uploaded_file = st.file_uploader("Upload PDF file", type="pdf")

embedding_function = SentenceTransformerEmbeddingFunction()

# Process PDF file if uploaded
if uploaded_file is not None:
    st.write("Loading PDF and creating Chroma collection...")
    st.write(f"Uploaded file: {uploaded_file.name}")

    # Read the uploaded file and extract text
    pdf_texts = _read_pdf(uploaded_file)

    # Load the Chroma collection using the read PDF text
    chroma_collection = load_chroma(pdf_texts, collection_name=uploaded_file.name, embedding_function=embedding_function)
    st.write(f"Loaded {chroma_collection.count()} chunks.")

   
# Query input
query = st.text_input('Enter your query:')

if query:
    # Perform document retrieval based on the query
    results = chroma_collection.query(query_texts=query, n_results=5, include=['documents', 'embeddings'])
    
    if results['documents']:
        retrieved_documents = results['documents'][0]

        st.write("Retrieved Documents:")
        # Display retrieved documents
        for document in retrieved_documents:
            st.write(word_wrap(document))

        # Embedding projection for query
        embeddings = chroma_collection.get(include=['embeddings'])['embeddings']
        umap_transform = umap.UMAP(random_state=0, transform_seed=0).fit(embeddings)

        # Visualize embeddings for the retrieved results
        query_embedding = embedding_function.embed([query])[0]
        retrieved_embeddings = results['embeddings'][0]
        projected_query_embedding = project_embeddings([query_embedding], umap_transform)
        projected_retrieved_embeddings = project_embeddings(retrieved_embeddings, umap_transform)

        st.write("Visualizing query embedding and retrieved embeddings...")

        # Query expansion for better retrieval
        expanded_query = expand_query_with_answer(query)
        st.write(f"Expanded Query: {expanded_query}")

        # Perform retrieval with the expanded query
        expanded_results = chroma_collection.query(query_texts=expanded_query, n_results=5, include=['documents', 'embeddings'])
        retrieved_documents = expanded_results['documents'][0]

        st.write("Documents after Query Expansion:")
        for doc in retrieved_documents:
            st.write(word_wrap(doc))

        # Multi-query expansion to cover different aspects
        augmented_queries = augment_multiple_query(query)
        queries = [query] + augmented_queries

        # Perform multi-query retrieval
        multi_query_results = chroma_collection.query(query_texts=queries, n_results=5, include=['documents', 'embeddings'])
        multi_retrieved_documents = multi_query_results['documents']

        # Deduplicate and display results
        unique_documents = set()
        st.write("Multi-query Results:")
        for documents in multi_retrieved_documents:
            for document in documents:
                unique_documents.add(document)

        for document in unique_documents:
            st.write(word_wrap(document))

        # Re-ranking documents using CrossEncoder
        st.write("Re-ranking documents for improved relevance...")
        cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        ranked_docs = rank_documents(cross_encoder, query, multi_retrieved_documents[0])

        st.write("Ranked Documents:")
        for rank, doc in ranked_docs.items():
            st.write(f"Rank {rank + 1}:")
            st.write(word_wrap(doc))
