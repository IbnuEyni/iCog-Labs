from django.http import JsonResponse
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from django.conf import settings
import os
from pypdf import PdfReader
import re
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
import google.generativeai as genai

def load_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def split_text(text, chunk_size=10000, chunk_overlap=500):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) < chunk_size:
            current_chunk += " " + sentence
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

class GeminiEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        gemini_api_key = settings.GEMINI_API_KEY
        if not gemini_api_key:
            raise ValueError("Gemini API Key not provided.")
        genai.configure(api_key=gemini_api_key)
        model = "models/embedding-001"
        title = "Custom query"
        return genai.embed_content(model=model, content=input, task_type="retrieval_document", title=title)["embedding"]

def create_or_load_chroma_db(documents, path, name):
    chroma_client = chromadb.PersistentClient(path=path)
    existing_collections = chroma_client.list_collections()
    collection_names = [collection.name for collection in existing_collections]

    if name in collection_names:
        db = chroma_client.get_collection(name=name, embedding_function=GeminiEmbeddingFunction())
    else:
        db = chroma_client.create_collection(name=name, embedding_function=GeminiEmbeddingFunction())
        for i, d in enumerate(documents):
            db.add(documents=d, ids=str(i))
    
    return db, name

def get_relevant_passage(query, db, n_results=1):
    results = db.query(query_texts=[query], n_results=n_results)
    passage = results['documents'][0] if results['documents'] else ""
    return passage

def generate_response(query, relevant_passages):
    gemini_api_key = settings.GEMINI_API_KEY
    if not gemini_api_key:
        raise ValueError("Gemini API Key not provided.")
    
    genai.configure(api_key=gemini_api_key)
    model = genai.GenerativeModel('gemini-pro')
    
    context = "\n\n".join(relevant_passages)
    prompt_template = f"""
    Answer the question as detailed as possible from the provided context,
    make sure to provide all the details. If the answer is not in the provided context, 
    just say, "The answer is not available in the context."
    
    Context:
    {context}
    
    Question:
    {query}
    
    Answer:
    """
    
    try:
        answer = model.generate_content(prompt_template)
        if hasattr(answer, 'text') and answer.text:
            return answer.text
        else:
            return "No answer available. Please check the input or try again."
    except Exception as e:
        print(f"Error generating response: {e}")
        return "An error occurred while generating the response."


def main(request):
    if request.method == 'POST':
        if 'pdf' in request.FILES:
            print("PDF file upload detected.")

            # Handle file upload
            uploaded_file = request.FILES['pdf']
            file_name = uploaded_file.name
            file_path = os.path.join(settings.MEDIA_ROOT, 'data', file_name)
            print("Saving PDF to:", file_path)
            
            # Saving the uploaded PDF file
            fs = FileSystemStorage(location=os.path.dirname(file_path))
            fs.save(file_name, uploaded_file)
            print("PDF file saved successfully.")
            
            # Store file path and collection name in session
            request.session['file_path'] = file_path
            request.session['db_name'] = os.path.splitext(file_name)[0]

            return JsonResponse({'success': True})

        elif 'query' in request.POST:
            print("Question submission detected.")

            # Handle question submission
            user_query = request.POST.get('query')
            print("Received question:", user_query)
            file_path = request.session.get('file_path')
            collection_name = request.session.get('db_name')
            
            if file_path and os.path.exists(file_path):
                pdf_text = load_pdf(file_path)
                text_chunks = split_text(pdf_text)
                db, _ = create_or_load_chroma_db(documents=text_chunks, path="chroma_db", name=collection_name)
                
                relevant_passages = get_relevant_passage(user_query, db, n_results=3)
                answer = generate_response(user_query, relevant_passages)
                print("Generated answer:", answer)
                
                return JsonResponse({'answer': answer})
            else:
                return JsonResponse({'error': 'No file uploaded or file path is invalid.'})

    # GET request or initial page load
    return render(request, 'upload.html')
