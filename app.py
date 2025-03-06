import streamlit as st
from PyPDF2 import PdfReader
import pickle
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
import os
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
# from langchain_pinecone import PineconeVectorStore
# from pinecone import Pinecone


load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001",google_api_key=os.getenv("GEMINI_API_KEY"));

# pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
# index_name = "rag-app-index"
# index = pc.Index(index_name)

# vector_store = PineconeVectorStore(embedding=embeddings, index=index)

generation_config = {
  "temperature": 1,
  "top_p": 0.95,
  "top_k": 40,
  "max_output_tokens": 8192,
  "response_mime_type": "text/plain",
}

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
        Assume yourself as a famous scientist name ALbert einstein. Your task is to answer the users question as if you are Einstein.
        You can refer this context to answer the user query. The context is {context}. Based on this context answer the query is {user_prompt}
        """
        response = chat_session.send_message(prompt)
        return response.text

    except Exception as e:
        st.error(f"Error generating text: {str(e)}")
        return None

# # Create a sidebar
# with st.sidebar:
#     st.title("PDF Chat APP")


def main():
    st.header("Chat with Pdf 💬")
    pdf = st.file_uploader("Upload your pdf", type="pdf")
    
    if pdf is not None:
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
        store_name = pdf.name[:-4]

        vector_store = None

        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl","rb") as f:
                vector_store = pickle.load(f)
        else:
            vector_store = InMemoryVectorStore(embeddings)
            documents = []
            for index,chunk in enumerate(chunks):
                document = Document(id=index,page_content=chunk)
                documents.append(document)
            
            vector_store.add_documents(documents=documents)

            with open(f"{store_name}.pkl","wb") as f:
                pickle.dump(vector_store, f)
        
        query = st.text_input("Ask question about PDF file")

        st.write(query)

        if query:
            results = vector_store.similarity_search(query=query,k=1)
            context = ""
            for doc in results:
                context += doc.page_content
                # print(f"* {doc.page_content} [{doc.metadata}]")
            
            response = get_gemini_response(query,context,"gemini-1.5-pro")
            st.write(response)


if __name__ == '__main__':
    main()