import streamlit as st
import time
import numpy as np
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
import google.generativeai as genai
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings


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
        Assume yourself as a person with name {st.session_state.avatar_selected}. Your task is to answer the users question as if you are {st.session_state.avatar_selected}.
        You can refer this context to answer the user query. The context is {context}. Based on this context answer the query is {user_prompt}
        """
        response = chat_session.send_message(prompt)
        return response.text

    except Exception as e:
        st.error(f"Error generating text: {str(e)}")
        return None


st.set_page_config(page_title="Chat with Avatar", page_icon="🤖")

st.markdown("# Chat With Avatar")
st.write(
    """Enjoy chatting with different famous avatars that you get inspired from most. This helps you to have a good interaction with
    people with real time behaviour. """
)

personalities = ["Albert Einstein", "MS Dhoni", "Elon Musk"]
avatar_images = {
    "Albert Einstein": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d3/Albert_Einstein_Head.jpg/800px-Albert_Einstein_Head.jpg",
    "MS Dhoni": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d5/MS_Dhoni_%28Prabhav_%2723_-_RiGI_2023%29.jpg/220px-MS_Dhoni_%28Prabhav_%2723_-_RiGI_2023%29.jpg",
    "Elon Musk": "https://upload.wikimedia.org/wikipedia/commons/thumb/3/34/Elon_Musk_Royal_Society_%28crop2%29.jpg/800px-Elon_Musk_Royal_Society_%28crop2%29.jpg",
}

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Albert Einstein")
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/d/d3/Albert_Einstein_Head.jpg/800px-Albert_Einstein_Head.jpg", width=200)

with col2:
    st.subheader("MS Dhoni")
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/d/d5/MS_Dhoni_%28Prabhav_%2723_-_RiGI_2023%29.jpg/220px-MS_Dhoni_%28Prabhav_%2723_-_RiGI_2023%29.jpg", width=200)

with col3:
    st.subheader("Elon Musk")
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/34/Elon_Musk_Royal_Society_%28crop2%29.jpg/800px-Elon_Musk_Royal_Society_%28crop2%29.jpg", width=200)


if "avatar_selected" not in st.session_state:
    st.session_state.avatar_selected = False

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if col1.button("Chat with Albert Einstein"):
    st.session_state.avatar_selected = "Albert Einstein"
    st.session_state.chat_history = []
    st.session_state.vector_store = PineconeVectorStore(embedding=embeddings, index=pc.Index("einstein-albert"))

if col2.button("Chat with MS Dhoni"):
    st.session_state.avatar_selected = "MS Dhoni"
    st.session_state.chat_history = []
    st.session_state.vector_store = PineconeVectorStore(embedding=embeddings, index=pc.Index("dhoni-success-story"))

if col3.button("Chat with Elon Musk"):
    st.session_state.avatar_selected = "Elon Musk"
    st.session_state.chat_history = []
    st.session_state.vector_store = PineconeVectorStore(embedding=embeddings, index=pc.Index("elon-musk"))

if st.session_state.avatar_selected:
    st.empty()
    st.write(f"You are now chatting with {st.session_state.avatar_selected}")
    
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Type your message here..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar=avatar_images[st.session_state.avatar_selected]):
            message_placeholder = st.empty()
            full_response = ""
            with st.spinner("Thinking..."):
                results = st.session_state.vector_store.similarity_search(query=prompt,k=1)

                context = ""
                for doc in results:
                    context += doc.page_content            
                assistant_response = get_gemini_response(prompt,context,"gemini-1.5-pro")
            for chunk in assistant_response.split():
                full_response += chunk + " "
                time.sleep(0.05)
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        st.session_state.chat_history.append({"role": "assistant", "content": full_response})
