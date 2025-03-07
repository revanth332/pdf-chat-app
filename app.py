import streamlit as st
from util import get_gemini_response,delete_all_indexes,create_vector_store

def main():

    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None

    st.header("Chat with Pdf 💬")
    pdf = st.file_uploader("Upload your pdf", type="pdf")

    if st.button("Delete All Indexes"):
        delete_all_indexes()
    
    if pdf is not None:
        index_name = pdf.name[:-4]

        with st.spinner("Processing PDF..."):
            st.session_state.vector_store = create_vector_store(index_name,pdf)
        
        query = st.text_input("Ask question about PDF file")
        if query:
            st.empty()
            results = st.session_state.vector_store.similarity_search(query=query,k=1)
            context = ""
            for doc in results:
                context += doc.page_content
            
            with st.spinner("Generating response..."):
                response = get_gemini_response("pdf-chat",context,query,"gemini-1.5-pro")
                st.write(response)


if __name__ == '__main__':
    main()