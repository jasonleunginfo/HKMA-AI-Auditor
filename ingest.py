import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

# Paste your OpenAI Key here
os.environ["OPENAI_API_KEY"] = "PASTE_YOUR_OPENAI_KEY_HERE"

def ingest_docs():
    loader = PyPDFLoader("TM-G-1.pdf")
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(data)
    
    # Using OpenAI Embeddings (Professional Standard)
    vectorstore = Chroma.from_documents(
        documents=chunks, 
        embedding=OpenAIEmbeddings(),
        persist_directory="./db"
    )
    print(f"Successfully ingested {len(chunks)} chunks using OpenAI.")

if __name__ == "__main__":
    ingest_docs()
