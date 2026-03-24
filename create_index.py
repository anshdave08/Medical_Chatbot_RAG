import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()
DATA_PATH = "Data"

def create_vector_store():
    documents = []

    print("📄 Loading PDFs...")

    for file in os.listdir(DATA_PATH):
        if file.endswith(".pdf"):
            pdf_path = os.path.join(DATA_PATH, file)
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            documents.extend(docs)

            print(f"Loaded {file}")

    print("Total pages:", len(documents))
    print("✂ Splitting text into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    docs = splitter.split_documents(documents)

    print("🧠 Creating embeddings (FREE HuggingFace model)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    print("📦 Creating FAISS vector store...")
    vectorstore = FAISS.from_documents(docs, embeddings)

    vectorstore.save_local("faiss_index")

    print("✅ FAISS index created successfully!")

if __name__ == "__main__":
    create_vector_store()