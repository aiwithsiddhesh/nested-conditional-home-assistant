from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from app.config import settings

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_document(path: str | Path) -> str:
    path = Path(path)

    if path.suffix == ".txt":
        return path.read_text(encoding="utf-8")

    if path.suffix == ".pdf":
        reader = PdfReader(path)
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    raise ValueError(f"Unsupported file type: {path.suffix}")


def build_retriever(path: str | Path) -> VectorStoreRetriever:
    text = load_document(path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    chunks = splitter.split_text(text)

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vector_store = FAISS.from_texts(chunks, embeddings)

    return vector_store.as_retriever(search_kwargs={"k": settings.retriever_k})
