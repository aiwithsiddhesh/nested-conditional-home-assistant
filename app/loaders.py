from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings
from app.state import HomeAssistantState

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_document(path: str | Path) -> list[Document]:
    path = Path(path)

    if path.suffix == ".txt":
        return [Document(page_content=path.read_text(encoding="utf-8"), metadata={"source": str(path)})]

    if path.suffix == ".pdf":
        return PyPDFLoader(str(path)).load()

    raise ValueError(f"Unsupported file type: {path.suffix}")


def build_retriever(path: str | Path) -> VectorStoreRetriever:
    documents = load_document(path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vector_store = FAISS.from_documents(chunks, embeddings)

    return vector_store.as_retriever(search_kwargs={"k": settings.retriever_k})


def format_context(documents: list[Document]) -> str:
    formatted = []
    for doc in documents:
        page_label = doc.metadata.get("page_label")
        prefix = f"[page {page_label}] " if page_label else ""
        formatted.append(f"{prefix}{doc.page_content}")
    return "\n\n".join(formatted)


def make_rag_node(retriever: VectorStoreRetriever, extra_fields: dict | None = None):
    def rag_node(state: HomeAssistantState) -> dict:
        latest_message = state["messages"][-1].content
        docs = retriever.invoke(latest_message)
        return {"retrieved_context": format_context(docs), **(extra_fields or {})}

    return rag_node
