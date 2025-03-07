import streamlit as st
import time
from util import create_vector_store,get_gemini_response

st.set_page_config(page_title="Chat with Avatar", page_icon="🤖")

st.markdown("# Chat With Avatar")
st.write(
    """Enjoy chatting with different famous avatars that you get inspired from most. This helps you to have a good interaction with
    people with real time behaviour. """
)

# personalities = ["Albert Einstein", "MS Dhoni", "Elon Musk"]
# avatar_images = {
#     "Albert Einstein": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d3/Albert_Einstein_Head.jpg/800px-Albert_Einstein_Head.jpg",
#     "MS Dhoni": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d5/MS_Dhoni_%28Prabhav_%2723_-_RiGI_2023%29.jpg/220px-MS_Dhoni_%28Prabhav_%2723_-_RiGI_2023%29.jpg",
#     "Elon Musk": "https://upload.wikimedia.org/wikipedia/commons/thumb/3/34/Elon_Musk_Royal_Society_%28crop2%29.jpg/800px-Elon_Musk_Royal_Society_%28crop2%29.jpg",
# }

avatars = [
    {
        "name": "Albert Einstein",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d3/Albert_Einstein_Head.jpg/800px-Albert_Einstein_Head.jpg",
        "index_name": "einstein-albert",
    },
    {
        "name": "MS Dhoni",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d5/MS_Dhoni_%28Prabhav_%2723_-_RiGI_2023%29.jpg/220px-MS_Dhoni_%28Prabhav_%2723_-_RiGI_2023%29.jpg",
        "index_name": "dhoni-success-story",
    },
    {
        "name": "Elon Musk",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/3/34/Elon_Musk_Royal_Society_%28crop2%29.jpg/800px-Elon_Musk_Royal_Society_%28crop2%29.jpg",
        "index_name": "elon-musk",
    },
    # Add more avatars here if needed
]

cols = st.columns(len(avatars))

for i, avatar in enumerate(avatars):
    with cols[i]:
        st.subheader(avatar["name"])
        st.image(avatar["image_url"], width=200)
        if st.button(f"Chat with {avatar['name']}"):
            st.session_state.avatar_selected = avatar["name"]
            st.session_state.chat_history = []
            st.session_state.vector_store = create_vector_store(avatar["index_name"])
            st.rerun()

if "avatar_selected" not in st.session_state:
    st.session_state.avatar_selected = False

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# if col1.button("Chat with Albert Einstein"):
#     st.session_state.avatar_selected = "Albert Einstein"
#     st.session_state.chat_history = []
#     st.session_state.vector_store = create_vector_store("einstein-albert")

# if col2.button("Chat with MS Dhoni"):
#     st.session_state.avatar_selected = "MS Dhoni"
#     st.session_state.chat_history = []
#     st.session_state.vector_store = create_vector_store("dhoni-success-story")

# if col3.button("Chat with Elon Musk"):
#     st.session_state.avatar_selected = "Elon Musk"
#     st.session_state.chat_history = []
#     st.session_state.vector_store = create_vector_store("elon-musk")

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

        with st.chat_message("assistant", avatar=next((a['image_url'] for a in avatars if a['name'] == st.session_state.avatar_selected),None)):
            message_placeholder = st.empty()
            full_response = ""
            with st.spinner("Thinking..."):
                results = st.session_state.vector_store.similarity_search(query=prompt,k=1)

                context = ""
                for doc in results:
                    context += doc.page_content            
                assistant_response = get_gemini_response("avatar-chat",context,prompt,"gemini-1.5-pro",avatar=st.session_state.avatar_selected)
            for chunk in assistant_response.split():
                full_response += chunk + " "
                time.sleep(0.05)
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        st.session_state.chat_history.append({"role": "assistant", "content": full_response})
