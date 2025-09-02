import streamlit as st
from rag_bot import load_pdf, split_documents, create_vectorstore, create_rag_chain, \
    create_bm25_index, bm25_search, save_faiss_index, load_faiss_index, load_bm25_index
import tempfile
import os

st.set_page_config(page_title="RAG Chatbot", page_icon="🤖")
st.title("BAJRA LEAVE POLICY")

# -----------------------------
# Session State
# -----------------------------
if "vectorstore" not in st.session_state:
    if os.path.exists("faiss_index"):
        st.session_state.vectorstore = load_faiss_index()
    else:
        st.session_state.vectorstore = None

if "qa_chain" not in st.session_state:
    if st.session_state.vectorstore:
        st.session_state.qa_chain = create_rag_chain(st.session_state.vectorstore)
    else:
        st.session_state.qa_chain = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "bm25" not in st.session_state or "bm25_docs" not in st.session_state:
    if os.path.exists("bm25_index.pkl"):
        st.session_state.bm25, st.session_state.bm25_docs = load_bm25_index()
    else:
        st.session_state.bm25 = None
        st.session_state.bm25_docs = None

# -----------------------------
# Sidebar: PDF Upload
# -----------------------------
with st.sidebar:
    st.header("Upload your PDF")
    uploaded_file = st.file_uploader("", type=["pdf"])
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            docs = load_pdf(tmp.name)
        split_docs = split_documents(docs)
        st.session_state.vectorstore = create_vectorstore(split_docs)
        st.session_state.qa_chain = create_rag_chain(st.session_state.vectorstore)
        save_faiss_index(st.session_state.vectorstore)
        st.session_state.bm25, st.session_state.bm25_docs = create_bm25_index(split_docs)
        st.success("PDF processed and FAISS index saved successfully!")

# -----------------------------
# Main: User Query
# -----------------------------
user_input = st.text_input("Ask anything about your document:")

if user_input and st.session_state.qa_chain:
    # BM25 reranking
    top_docs = bm25_search(user_input, st.session_state.bm25, st.session_state.bm25_docs, top_k=5)
    qa_chain = st.session_state.qa_chain
    response = qa_chain.run(user_input)
    st.session_state.chat_history.append({"user": user_input, "bot": response})



for chat in st.session_state.chat_history:
    # Human bubble (right-aligned)
    st.markdown(
        f"""
        <div style='display:flex; justify-content:flex-end; margin:5px'>
            <div style='background-color:#DCF8C6; padding:10px; border-radius:10px; max-width:70%'>
                <b>Human:</b> {chat['user']}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # AI bubble (left-aligned)
    st.markdown(
        f"""
        <div style='display:flex; justify-content:flex-start; margin:5px'>
            <div style='background-color:#F1F0F0; padding:10px; border-radius:10px; max-width:70%'>
                <b>AI:</b> {chat['bot']}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# Close scrollable container
st.markdown("</div>", unsafe_allow_html=True)

