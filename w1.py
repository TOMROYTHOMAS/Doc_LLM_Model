import streamlit as st
import os
import chromadb
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.docstore.document import Document

# ✅ Use PersistentClient without specifying Settings
chroma_client = chromadb.PersistentClient(path="db")  # ChromaDB automatically manages DB selection

def process_document(uploaded_file, user_question):
    file_extension = uploaded_file.name.split(".")[-1].lower()
    file_path = os.path.join("uploaded." + file_extension)

    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    docs = []
    if file_extension == "pdf":
        loader = PyPDFLoader(file_path)
        docs = loader.load()
    elif file_extension == "txt":
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
            docs = [Document(page_content=text)]

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = text_splitter.split_documents(docs)

    # ✅ Create and persist vector store
    vector_store = Chroma.from_documents(
        split_docs,
        HuggingFaceEmbeddings(),
        persist_directory="db"
    )
    vector_store.persist()

    llm = Ollama(model="llama3.2")

    # Define RAG-based retrieval function
    def create_rag():
        return RetrievalQA.from_llm(llm, retriever=vector_store.as_retriever())

    # Create the RAG-based Q&A chain
    qa_chain = create_rag()

    if user_question:
        response = qa_chain.run(user_question)
        return response
    else:
        return "Please ask a question about the document."

# Streamlit UI
st.title("RAG-based Q&A Chatbot with Ollama")
st.markdown("Upload a PDF or TXT document and ask a question. The system will search for answers based on the content of the document.")

# File uploader
uploaded_file = st.file_uploader("Upload Document (PDF/TXT)", type=["pdf", "txt"])

# Textbox for asking questions
user_question = st.text_input("Ask a question about the document:")

if uploaded_file and user_question:
    response = process_document(uploaded_file, user_question)
    st.write("### Answer:")
    st.write(response)
elif uploaded_file:
    st.write("Please enter a question to get answers from the document.")
