import shutil
import sys
from pathlib import Path

from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFDirectoryLoader,
    TextLoader,
)
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHUNK_OVERLAP, CHUNK_SIZE, EMBEDDING_MODEL, KNOWLEDGE_BASE_DIR


def load_folder(folder: str) -> list[Document]:
    """Load all PDF, TXT and Markdown files in the folder, tagged with their file name."""
    documents = PyPDFDirectoryLoader(folder, recursive=True).load()
    documents += DirectoryLoader(
        folder, glob=["*.txt", "*.md"], recursive=True, loader_cls=TextLoader
    ).load()
    for page in documents:
        page.metadata["source"] = Path(page.metadata["source"]).name
    return documents


if __name__ == "__main__":
    pages = load_folder(sys.argv[1])
    shutil.rmtree(KNOWLEDGE_BASE_DIR, ignore_errors=True)
    Chroma.from_documents(
        RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        ).split_documents(pages),
        HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL),  # type: ignore
        persist_directory=KNOWLEDGE_BASE_DIR,
        collection_metadata={"hnsw:space": "cosine"},
    )
    print(f"Ingested {sorted({p.metadata['source'] for p in pages})}")
