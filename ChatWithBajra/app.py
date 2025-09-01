import os
import streamlit as st
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory

# Page Configuration

st.set_page_config(page_title="Bajra Leave Policy Assistant", page_icon="📄", layout="wide")
st.title("📄 Bajra Leave Policy Assistant")

# Directory for uploaded PDFs
PDF_STORAGE_PATH = "document_store/PDFs/"
os.makedirs(PDF_STORAGE_PATH, exist_ok=True)

# Models & Vector DB
embedding_model = OllamaEmbeddings(model="llama3.2")
vector_db = InMemoryVectorStore(embedding_model)
llm = OllamaLLM(model="llama3.2", temperature=0.7)

# Prompt Template
PROMPT_TEMPLATE = """
You are a helpful Assistant. Use the given context and previous conversation history to answer the query. 
If you don't know the answer, state that you don't know. Be concise and factual.

Chat History:
{chat_history}

Query: {user_query}

Context: {document_context}

Answer:
"""

# Memory Initialization
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

if "messages" not in st.session_state:
    st.session_state.messages = []  


# Helper Functions

def save_uploaded_pdf(uploaded_file):
    file_path = os.path.join(PDF_STORAGE_PATH, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def load_pdf(file_path):
    loader = PDFPlumberLoader(file_path)
    return loader.load()

def chunk_documents(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(docs)

def index_chunks(chunks):
    vector_db.add_documents(chunks)

def find_similar_docs(query):
    return vector_db.similarity_search(query)

def generate_answer(query, docs):
    context = "\n\n".join([doc.page_content for doc in docs])
    chat_history = st.session_state.memory.load_memory_variables({}).get("chat_history", "")
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    chain = prompt | llm
    response = chain.invoke({
        "user_query": query,
        "document_context": context,
        "chat_history": chat_history
    })
    st.session_state.memory.save_context({"input": query}, {"output": response})
    return response


# Streamlit App UI

st.sidebar.header("Upload PDF Document")
uploaded_pdf = st.sidebar.file_uploader("Upload a PDF", type="pdf")

if uploaded_pdf:
    with st.spinner("Processing PDF..."):
        pdf_path = save_uploaded_pdf(uploaded_pdf)
        documents = load_pdf(pdf_path)
        chunks = chunk_documents(documents)
        index_chunks(chunks)
    st.sidebar.success("✅ Document Processed Successfully!")

    st.subheader("💬 Ask Questions About the Policy")

    # Display previous messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_query = st.chat_input("Ask your query about Bajra Leave Policy")

    if user_query:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.spinner("Analyzing the documents..."):
            similar_docs = find_similar_docs(user_query)
            answer = generate_answer(user_query, similar_docs)

        # Add assistant response
        st.session_state.messages.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.markdown(answer)

else:
    st.info("Please upload a PDF to start.")
