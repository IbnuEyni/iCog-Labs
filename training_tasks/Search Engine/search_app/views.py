from django.http import HttpResponseRedirect, JsonResponse
from django.shortcuts import get_object_or_404, render, redirect
from .models import Document
import re
import string
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from django.views.decorators.csrf import csrf_exempt
import PyPDF2
import docx

# Preprocess and clean data
nltk.download('stopwords')
nltk.download("wordnet")
nltk.download('punkt')

stopwords_list = stopwords.words('english')

english_stopset = set(stopwords.words('english')).union(
    {"things", "that's", "something", "take", "don't", "may", "want", "you're",
     "set", "might", "says", "including", "lot", "much", "said", "know",
     "good", "step", "often", "going", "thing", "things", "think",
     "back", "actually", "better", "look", "find", "right", "example",}
)

lemmatizer = WordNetLemmatizer()

def preprocess_data(docs):
    documents_clean = []
    for d in docs:
        document_test = re.sub(r'[^\x00-\x7F]+', ' ', d)  # Replace non-ASCII characters
        document_test = re.sub(r'@\w+', '', document_test)  # Remove mentions
        document_test = document_test.lower()  # Convert to lower case
        document_test = re.sub(r'[%s]' % re.escape(string.punctuation), ' ', document_test)  # Remove punctuation
        document_test = re.sub(r'[0-9]', '', document_test)  # Remove numbers
        document_test = re.sub(r'\s{2,}', ' ', document_test)  # Remove multiple spaces
        documents_clean.append(document_test)

    processed_docs = [' '.join([lemmatizer.lemmatize(word) for word in text.split()]) for text in documents_clean]
    return processed_docs

def upload_document_page(request):
    return render(request, 'upload_document.html')

@csrf_exempt
def upload_and_process_documents(request):
    if request.method == 'POST':
        if 'document' in request.FILES:
            file = request.FILES['document']
            file_extension = file.name.split('.')[-1].lower()
            print(f"[INFO] Uploading document: {file.name} with extension: {file_extension}")  

            if file_extension == 'pdf':
                text = extract_text_from_pdf(file)
            elif file_extension == 'docx':
                text = extract_text_from_docx(file)
            elif file_extension == 'txt':
                text = extract_text_from_txt(file)
            else:
                print("[ERROR] Unsupported file format")  
                return JsonResponse({'error': 'Unsupported file format'}, status=400)

            # Save document to database
            Document.objects.create(title=file.name, file_type=file_extension, content=text)
            print(f"[INFO] Document saved: {file.name}") 

            return redirect('query_page')
        else:
            print("[ERROR] No file uploaded") 
            return JsonResponse({'error': 'No file uploaded'}, status=400)
    else:
        print("[ERROR] Invalid request method")  
        return JsonResponse({'error': 'Invalid request method'}, status=405)

def query_page(request):
    return render(request, 'query_page.html')

@csrf_exempt
def process_query(request):
    if request.method == 'POST':
        query = request.POST.get('query', '')
        n = int(request.POST.get('n', 1))  
        print(f"[INFO] Processing query: {query}") 

        # Retrieve all document contents and titles from the database
        documents = Document.objects.all()

        if not documents:
            print("[ERROR] No documents available") 
            return JsonResponse({'error': 'No documents available'}, status=400)

        # extract document texts and titles
        doc_texts = [doc.content for doc in documents]
        titles = [doc.title for doc in documents]

        # Preprocess document texts
        new_docs = preprocess_data(doc_texts)
        global new_titles
        new_titles = [" ".join([lemmatizer.lemmatize(doc) for doc in text.split(" ")]) for text in titles]

        # Initialize TfidfVectorizer
        global vectorizer
        vectorizer = TfidfVectorizer(analyzer='word',
                                     ngram_range=(1, 2),
                                     min_df=0.002,
                                     max_df=0.99,
                                     max_features=1000,
                                     lowercase=True,
                                     stop_words=stopwords_list)

        # Fit and transform documents
        X = vectorizer.fit_transform(new_docs)
        df = pd.DataFrame(X.T.toarray()) 

        # Query processing
        lemma_ops = ' '.join([lemmatizer.lemmatize(word) for word in nltk.word_tokenize(query)])
        print(f"[INFO] Lemmatized query: {lemma_ops}") 

        result = get_similar_articles(lemma_ops, new_docs, df, new_titles, n)
        print(f"[INFO] Query results: {result}")  
        return JsonResponse({'result': result})
    else:   
        print("[ERROR] Invalid request method") 
        return JsonResponse({'error': 'Invalid request method'}, status=405)


def get_similar_articles(query, doc_texts, df, titles, n):
    # Transform the query to the vector space
    q_vec = vectorizer.transform([query]).toarray().reshape(df.shape[0],)
    print(f"Query vector: {q_vec}")
    
    # Calculate similarity
    sim = {}
    for i in range(len(doc_texts)):
        # Calculate norms (magnitudes)
        doc_norm = np.linalg.norm(df.loc[:, i])
        print(f"Document vector for doc {i}: {df.loc[:, i]}")
        
        query_norm = np.linalg.norm(q_vec)
        
        # Ensure that neither norm is zero before dividing
        if doc_norm == 0 or query_norm == 0:
            sim[i] = 0  # Set similarity to 0 if either vector has zero magnitude
        else:

            sim[i] = np.dot(df.loc[:, i].values, q_vec) / (doc_norm * query_norm)   # Calculate the similarity

    # Sort the similarities and select the top n results
    sim_sorted = sorted(sim.items(), key=lambda x: x[1], reverse=True)[:n]
    print(sim_sorted)

    results = []
    for i, v in sim_sorted:
        # Extract the first sentence
        first_sentence = doc_texts[i].split(" ")[:10]
        first_sentence = " ".join(first_sentence)
        results.append({
            'title': titles[i].split(".")[0],
            'content': first_sentence.strip(),  # Ensure the sentence ends with a period
            'similarity': v
        })

    return results



# Extract text functions
def extract_text_from_pdf(file):
    try:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + " "
        print("[INFO] Extracted text from PDF")  
        return text
    except Exception as e:
        print(f"[ERROR] Error extracting PDF content: {e}") 
        return f"Error extracting PDF content: {e}"

def extract_text_from_docx(file):
    try:
        doc = docx.Document(file)
        text = "\n".join([para.text for para in doc.paragraphs])
        print("[INFO] Extracted text from DOCX")  
        return text
    except Exception as e:
        print(f"[ERROR] Error extracting DOCX content: {e}")  
        return f"Error extracting DOCX content: {e}"

def extract_text_from_txt(file):
    try:
        text = file.read().decode('utf-8')
        print("[INFO] Extracted text from TXT")  
        return text
    except Exception as e:
        print(f"[ERROR] Error extracting TXT content: {e}")  
        return f"Error extracting TXT content: {e}"

# @login_required
def delete_document(request, document_id):
    document = get_object_or_404(Document, id=document_id)
     
    # Delete the document
    document.delete()
    
    # Redirect to the document list page after deletion
    return HttpResponseRedirect('/documents/')

def document_list(request):
    # Get all documents from the database
    documents = Document.objects.all()

    # Pass the documents to the template
    return render(request, 'document_list.html', {'documents': documents})

def search_crawler(request):
    return render(request, 'query_crawler.html')