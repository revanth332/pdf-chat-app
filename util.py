import google.generativeai as genai
from dotenv import load_dotenv
import re
import time
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone,ServerlessSpec
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from uuid import uuid4

load_dotenv()
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001",google_api_key=os.getenv("GEMINI_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_KEY"),ssl_verify=False)

generation_config = {
  "temperature": 1,
  "top_p": 0.95,
  "top_k": 40,
  "max_output_tokens": 8192,
  "response_mime_type": "text/plain",
}

def sanitize_index_name(name):
    """Sanitizes a string to be a valid Pinecone index name."""
    name = name.lower()  # Convert to lowercase
    name = re.sub(r"[^a-z0-9-]", "-", name)  # Replace invalid chars with hyphens
    name = re.sub(r"-+", "-", name) # replace multiples - with a single -
    name = re.sub(r"^-|-$", "", name) # remove - at the start or end
    return name

def delete_all_indexes():
    """Deletes all Pinecone indexes."""
    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
    if existing_indexes:
        for index_name in existing_indexes:
            pc.delete_index(index_name)
    else:
        print("No indexes to delete.")
 
def pdf_to_vector_documents(pdf):
    pdf_reader = PdfReader(pdf)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 200,
        length_function = len
    )

    chunks = text_splitter.split_text(text=text)
    documents = []
    for index,chunk in enumerate(chunks):
        document = Document(page_content=chunk)
        documents.append(document)
    return documents

def create_vector_store(index_name,pdf = None):
    index_name = sanitize_index_name(index_name)
    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
    vector_store = None
    if index_name not in existing_indexes:
        pc.create_index(
            name=index_name,
            dimension=768,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        while not pc.describe_index(index_name).status["ready"]:
            time.sleep(1)
        
        vector_store_index = pc.Index(index_name)
        vector_store = PineconeVectorStore(embedding=embeddings, index=vector_store_index)
        vector_documents = pdf_to_vector_documents(pdf=pdf)
        uuids = [str(uuid4()) for _ in range(len(vector_documents))]
        vector_store.add_documents(documents=vector_documents,ids=uuids)
    else:
        vector_store = PineconeVectorStore(embedding=embeddings, index=pc.Index(index_name))
    
    return vector_store

def get_prompt(chat_type,context,user_query,avatar=None):
    prompt_switcher = {
        "pdf-chat" : f"""
                    You are AI assistant that responds to user queries with care and responsibility. Before answering a question you have to refer to the context given.
                    The context is '{context}'. The query is '{user_query}'. Now answer the question based on the given context.
                    """,
        "avatar-chat" : f"""
                    Assume yourself as a person with name '{avatar}'. Your task is to answer the users question as if you are '{avatar}'.
                    You can refer this context to answer the user query. The context is '{context}'. Based on this context answer the query is '{user_query}'
                    """
    }
    default_prompt = f"""You are a smart AI assistant that answers to the users query. The query is '{user_query}'"""

    return prompt_switcher.get(chat_type,default_prompt)

def get_gemini_response(chat_type,context,user_query,model_name,avatar=None):
    chat_session = None
    try:
        if chat_session is None:
            model = genai.GenerativeModel(
                model_name=model_name,
                generation_config=generation_config,
                system_instruction="system instruction"
            )
            chat_session = model.start_chat()

        prompt = get_prompt(chat_type,context,user_query,avatar)
        response = chat_session.send_message(prompt)

        return response.text

    except Exception as e:
        print(f"Error generating text: {str(e)}")
        return None
   