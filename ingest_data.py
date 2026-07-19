"""
ingest_data.py
Reads PDFs from data/documents/ and builds a FAISS vector database.
"""

import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def build_vector_db():
    print("Loading PDFs from data/documents/...")
    loader = PyPDFDirectoryLoader("./data/doucments/")
    documents = loader.load()

    if not documents:
        print("No PDFs found. Please add PDFs to data/documents/ folder.")
        return

    print(f"Loaded {len(documents)} pages. Splitting text...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)

    print("Generating embeddings using HuggingFace...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    print("Building FAISS Vector Database...")
    vector_db = FAISS.from_documents(chunks, embeddings)
    
    # Save database locally
    os.makedirs("data/vector_db", exist_ok=True)
    vector_db.save_local("data/vector_db")
    print("Vector Database saved successfully in data/vector_db/")

if __name__ == "__main__":
    build_vector_db()