from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

DATA_PATH = "data/"

def load_pdf_files(data_path):
    loader = DirectoryLoader(
        data_path,
        glob="*.pdf",
        loader_cls=PyPDFLoader,
    )
    documents = loader.load()
    return documents


documents = load_pdf_files(data_path=DATA_PATH)
print("Length of PDF pages:", len(documents))


def create_chunks(extracted_documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    text_chunks = text_splitter.split_documents(extracted_documents)
    return text_chunks


text_chunks = create_chunks(extracted_documents=documents)
print("Length of the Text chunks:", len(text_chunks))


def get_embedding_model():
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return embedding_model


embedding_model = get_embedding_model()

DB_FAISS_PATH = "vectorstore/db_faiss"
os.makedirs(os.path.dirname(DB_FAISS_PATH), exist_ok=True)
db = FAISS.from_documents(text_chunks, embedding_model)
db.save_local(DB_FAISS_PATH)