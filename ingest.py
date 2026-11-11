"""Load PDFs/TXT files from ./docs, chunk them, embed them, and save a FAISS index.

Run:  python ingest.py
"""
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag import INDEX_DIR, get_embeddings

DOCS_DIR = Path("docs")
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


def load_documents():
    pages, files = [], sorted(DOCS_DIR.glob("*"))
    used = 0
    for f in files:
        if f.suffix.lower() == ".pdf":
            loader = PyPDFLoader(str(f))
        elif f.suffix.lower() == ".txt":
            loader = TextLoader(str(f), encoding="utf-8")
        else:
            continue
        docs = loader.load()
        for d in docs:
            d.metadata["source_name"] = f.name
            # PyPDFLoader pages are 0-indexed; show human-friendly numbers
            d.metadata["page_label"] = d.metadata.get("page", 0) + 1
        pages.extend(docs)
        used += 1
    return pages, used


def main():
    pages, n_files = load_documents()
    if not pages:
        raise SystemExit("No PDF/TXT files found in ./docs. Add some and re-run.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(pages)

    print("Embedding chunks (first run downloads the model, be patient)...")
    store = FAISS.from_documents(chunks, get_embeddings())
    store.save_local(INDEX_DIR)

    print(f"\nDocuments: {n_files}")
    print(f"Pages:     {len(pages)}")
    print(f"Chunks:    {len(chunks)}  (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"Index saved to ./{INDEX_DIR}")
    print("Note these numbers, they go on your resume as [N].")


if __name__ == "__main__":
    main()
