import os
import pickle
import asyncio
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from rank_bm25 import BM25Okapi

# Set your Google API key
os.environ['GOOGLE_API_KEY'] = 'AIzaSyAsC36ctWxb_rNFyPGu5z3IgQsWbfZxWPU'

FAISS_INDEX_PATH = "faiss_index"
BM25_INDEX_PATH = "bm25_index.pkl"

# -----------------------------
# Document Handling
# -----------------------------
def load_pdf(file_path):
    loader = PyPDFLoader(file_path)
    return loader.load()

def split_documents(documents, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(documents)

# -----------------------------
# Vector Store + Embeddings (Async-safe)
# -----------------------------
async def create_embeddings():
    return GoogleGenerativeAIEmbeddings(model='models/embedding-001')

def create_vectorstore(documents):
    embeddings = asyncio.run(create_embeddings())
    vectorstore = FAISS.from_documents(documents, embeddings)
    return vectorstore

def load_faiss_index(index_path=FAISS_INDEX_PATH):
    embeddings = asyncio.run(create_embeddings())
    return FAISS.load_local(index_path, embeddings,allow_dangerous_deserialization=True)

def save_faiss_index(vectorstore, index_path=FAISS_INDEX_PATH):
    vectorstore.save_local(index_path)

# -----------------------------
# BM25 Index
# -----------------------------
def create_bm25_index(documents):
    corpus = [doc.page_content.split() for doc in documents]
    bm25 = BM25Okapi(corpus)
    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump((bm25, documents), f)
    return bm25, documents

def load_bm25_index():
    with open(BM25_INDEX_PATH, "rb") as f:
        return pickle.load(f)

def bm25_search(query, bm25, docs, top_k=5):
    tokenized_query = query.split()
    scores = bm25.get_scores(tokenized_query)
    top_indices = scores.argsort()[-top_k:][::-1]
    return [docs[i] for i in top_indices]

# -----------------------------
# RAG Chain with Memory (Async-safe)
# -----------------------------
async def create_chat_model():
    return ChatGoogleGenerativeAI(temperature=0,model='gemini-2.5-flash')

def create_rag_chain(vectorstore):
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    chat_model = asyncio.run(create_chat_model())
    qa_chain = ConversationalRetrievalChain.from_llm(
        chat_model,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return qa_chain
