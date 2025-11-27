import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DataFrameLoader
import os

# 1. Setup paths
DATA_PATH = "data/counseling.csv"
DB_PATH = "models/faiss_index"

def build_vector_db():
    print("⏳ Loading data... this might take a minute.")
    # Load dataset - adjusting for common column names in this dataset
    df = pd.read_csv(DATA_PATH)
    
    # We only need the 'Context' (User) and 'Response' (Therapist)
    # Check your csv column names! We assume 'Context' and 'Response' exist.
    # If your dataset uses 'question'/'answer', change them below.
    df = df[['Context', 'Response']].dropna()
    
    # Create the "Knowledge Base"
    # We load the 'Response' as the content we want to give back to the user
    loader = DataFrameLoader(df, page_content_column="Response")
    documents = loader.load()

    # 2. Convert text to numbers (Embeddings)
    print("🧠 Vectorizing text... (This allows the AI to understand meaning)")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # 3. Create and Save the Vector Database
    db = FAISS.from_documents(documents, embeddings)
    db.save_local(DB_PATH)
    print(f"✅ Database saved to {DB_PATH}")

def load_retriever():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    # Allow dangerous deserialization since we created the file ourselves
    db = FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)
    return db

if __name__ == "__main__":
    # Run this file directly to build the database once
    build_vector_db()