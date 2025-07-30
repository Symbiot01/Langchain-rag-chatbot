import streamlit as st
from rag_pipeline import build_rag_chain

st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title(" RAG Chatbot (Groq + LangChain)")

@st.cache_resource
def load_chain():
    return build_rag_chain()

qa_chain = load_chain()

user_query = st.text_input("💬 Ask a question based on the document:", placeholder="e.g., What is the main topic?")

if user_query:
    with st.spinner("Generating answer..."):
        result = qa_chain({"query": user_query})
        st.markdown("### ✅ Answer")
        st.write(result["result"])

        with st.expander("📄 Source Documents"):
            for doc in result["source_documents"]:
                st.write(doc.page_content)
