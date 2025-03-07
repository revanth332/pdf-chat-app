import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
import os
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone,ServerlessSpec
import time
from uuid import uuid4
import re

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
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

def get_gemini_response(user_prompt: str,context : str, model_name):
    chat_session = None
    try:
        if chat_session is None:
            model = genai.GenerativeModel(
                model_name=model_name,
                generation_config=generation_config,
                system_instruction="system instruction"
            )
            chat_session = model.start_chat()

        prompt = f"""
        You are AI assistant that responds to user queries with care and responsibility. Before answering a question you have to refer to the context given.
        The context is '{context}'. The query is '{user_prompt}'. Now answer the question based on the given context.
        """
        response = chat_session.send_message(prompt)
        return response.text

    except Exception as e:
        st.error(f"Error generating text: {str(e)}")
        return None

# # Create a sidebar
# with st.sidebar:
#     st.title("PDF Chat APP")

def delete_all_indexes():
    """Deletes all Pinecone indexes."""
    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
    if existing_indexes:
        st.write("Deleting all existing indexes...")
        for index_name in existing_indexes:
            pc.delete_index(index_name)
            st.write(f"Deleted index: {index_name}")
    else:
        st.write("No indexes to delete.")

def main():
    st.header("Chat with Pdf 💬")
    pdf = st.file_uploader("Upload your pdf", type="pdf")
    if st.button("Delete All Indexes"):
        delete_all_indexes()
    if pdf is not None:
        vector_store = None
        index_name = pdf.name[:-4]
        index_name = sanitize_index_name(index_name)

        existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
        st.write(existing_indexes)

        if index_name not in existing_indexes:
            st.write("creating new vector index")
            pc.create_index(
                name=index_name,
                dimension=768,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )

            while not pc.describe_index(index_name).status["ready"]:
                time.sleep(1)
            
            store_index = pc.Index(index_name)

            st.write("creating new vector store")
            vector_store = PineconeVectorStore(embedding=embeddings, index=store_index)
            with st.spinner("Processing PDF..."):
                st.write("Reading pages")
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
                
                uuids = [str(uuid4()) for _ in range(len(documents))]
                st.write("Adding documents to vector store")
                vector_store.add_documents(documents=documents,ids=uuids)
        else:
            st.write("pulling existing vector store")
            vector_store = PineconeVectorStore(embedding=embeddings, index=pc.Index(index_name))
        
        query = st.text_input("Ask question about PDF file")
        if query:
            results = vector_store.similarity_search(query=query,k=1)
            context = ""
            for doc in results:
                context += doc.page_content
            
            response = get_gemini_response(query,context,"gemini-1.5-pro")
            st.write(response)


if __name__ == '__main__':
    main()