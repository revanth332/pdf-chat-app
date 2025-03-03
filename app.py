import streamlit as st
from PyPDF2 import PdfReader
import pickle
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceInstructEmbeddings
# from google import genai
# from google.genai import types
import os

load_dotenv()

# client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

instructor_embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large")


# Create a sidebar
with st.sidebar:
    st.title("PDF Chat APP")


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

        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl","rb") as f:
                VectorStore = pickle.load(f)
        else:
            VectorStore = FAISS.from_texts(chunks, embedding=instructor_embeddings)
            with open(f"{store_name}.pkl","wb") as f:
                pickle.dump(VectorStore, f)

        st.write(chunks)


if __name__ == '__main__':
    main()