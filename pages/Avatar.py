import streamlit as st
import time
import numpy as np

st.set_page_config(page_title="Chat with Avatar", page_icon="🤖")

st.markdown("# Chat With Avatar")
st.write(
    """Enjoy chatting with different famous avatars that you get inspired from most. This helps you to have a good interaction with
    people with real time behaviour. """
)

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

if col1.button("Chat with Albert Einstein"):
    st.session_state.avatar_selected = "Albert Einstein"
    st.session_state.chat_history = []

if col2.button("Chat with MS Dhoni"):
    st.session_state.avatar_selected = "MS Dhoni"
    st.session_state.chat_history = []

if col3.button("Chat with Elon Musk"):
    st.session_state.avatar_selected = "Elon Musk"
    st.session_state.chat_history = []

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

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            assistant_response = "Hello, I am " + st.session_state.avatar_selected + " How can I help you?"
            for chunk in assistant_response.split():
                full_response += chunk + " "
                time.sleep(0.05)
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        st.session_state.chat_history.append({"role": "assistant", "content": full_response})
