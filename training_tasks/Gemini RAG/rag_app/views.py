import os
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import FileSystemStorage
from django.conf import settings
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from docx import Document

# Load environment variables
load_dotenv()

def initialize_components(file_path, file_type):
    # Ensure that file_path is an absolute path relative to the data directory
    absolute_file_path = os.path.join(settings.BASE_DIR, file_path)

    # Load the file data
    if file_type == 'pdf':
        loader = PyPDFLoader(absolute_file_path)
        data = loader.load()
    elif file_type == 'txt':
        with open(absolute_file_path, 'r', encoding='utf-8') as file:
            data = file.readlines()
    elif file_type in ['doc', 'docx']:
        data = []
        doc = Document(absolute_file_path)
        for para in doc.paragraphs:
            data.append(para.text)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")

    # Initialize the text splitter with chunk size and overlap
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
    
    # Split the loaded data into documents with overlap
    docs = text_splitter.split_documents(data)

    # Initialize embeddings with API key from settings
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=settings.GEMINI_API_KEY  # Use GEMINI_API_KEY here
        )
    )
    
    # Set up the retriever with similarity search
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

    # Initialize LLM with API key from settings
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        temperature=0,
        max_tokens=None,
        timeout=None,
        google_api_key=settings.GEMINI_API_KEY  # Use GEMINI_API_KEY here
    )

    return retriever, llm


@csrf_exempt
def chat_view(request):
    print("Entered chat_view function")  # Log entry into the function

    if request.method == 'POST':
        print("Handling POST request")  # Log that a POST request is being handled

        if 'pdf_file' in request.FILES or 'txt_file' in request.FILES or 'doc_file' in request.FILES:
            uploaded_file = request.FILES.get('pdf_file') or request.FILES.get('txt_file') or request.FILES.get('doc_file')
            file_type = 'pdf' if 'pdf_file' in request.FILES else 'txt' if 'txt_file' in request.FILES else 'doc' if uploaded_file.name.endswith('.doc') else 'docx'

            print(f"Received {file_type} file upload")  # Log that a file was received

            # Save the file to the 'data' directory
            fs = FileSystemStorage(location=os.path.join(settings.BASE_DIR, 'data'))
            file_name = fs.save(uploaded_file.name, uploaded_file)
            file_path = os.path.join('data', file_name)  # Use relative path from BASE_DIR
            print(f"File saved at: {file_path}")  # Log where the file was saved

            try:
                # Initialize components with the saved file path
                print("Initializing components with file")  # Log that component initialization is starting
                retriever, llm = initialize_components(file_path, file_type)
                
                # Store file path and type in session
                request.session['file_path'] = file_path
                request.session['file_type'] = file_type
                print("Components metadata stored in session successfully")  # Log successful storage

                return JsonResponse({"success": True, "message": f"{file_type.upper()} uploaded successfully"})
            except Exception as e:
                print(f"Error initializing components: {e}")  # Log any errors during initialization
                return JsonResponse({"success": False, "error": "Error initializing components"})

        elif 'query' in request.POST:
            query = request.POST.get('query')
            if query:
                print(f"Received query: {query}")  # Log the received query

                try:
                    # Retrieve components from metadata in session
                    file_path = request.session.get('file_path')
                    file_type = request.session.get('file_type')

                    if not file_path or not file_type:
                        print("Error: Metadata not available or expired")  # Log if metadata is not available
                        return JsonResponse({"answer": "Error: Metadata not available. Please upload a file first."})

                    # Reinitialize components with metadata 
                    retriever, llm = initialize_components(file_path, file_type)
                    print("Components reinitialized successfully")  # Log successful reinitialization

                    # Define the system prompt
                    system_prompt = (
                        "You are an assistant for question-answering tasks. "
                        "Use the following pieces of retrieved context to answer "
                        "the question. If you don't know the answer, say that you "
                        "don't know. Use three sentences maximum and keep the "
                        "answer concise."
                        "\n\n"
                        "{context}"
                    )

                    # Create the prompt template and the question-answer chain
                    prompt = ChatPromptTemplate.from_messages(
                        [
                            ("system", system_prompt),
                            ("human", "{input}"),
                        ]
                    )
                    print("Created ChatPromptTemplate")  # Log the creation of the prompt template

                    question_answer_chain = create_stuff_documents_chain(llm, prompt)
                    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
                    print("Created RAG chain")  # Log the creation of the RAG chain

                    # Generate the response
                    response = rag_chain.invoke({"input": query})
                    print(f"Response generated: {response['answer']}")  # Log the generated response

                    return JsonResponse({"answer": response["answer"]})
                except Exception as e:
                    print(f"Error during query processing: {e}")  # Log any errors during query processing
                    return JsonResponse({"answer": "Error processing query. Please check the server logs."})

    print("Rendering chat.html")  # Log that the chat.html template is being rendered
    return render(request, 'chat.html')