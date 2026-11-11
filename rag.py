"""Core RAG logic shared by the app and the evaluation script."""
import os
import time

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

INDEX_DIR = "faiss_index"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 4

PROMPT = """You are a helpful assistant that answers questions using ONLY the context below.
If the answer is not in the context, say "I couldn't find that in the provided documents."
Do not make things up. Be concise and cite sources like [1], [2] where relevant.

Context:
{context}

Question: {question}

Answer:"""


def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )


def load_store():
    return FAISS.load_local(
        INDEX_DIR, get_embeddings(), allow_dangerous_deserialization=True
    )


def get_llm():
    """Pick the LLM from the LLM_PROVIDER env var: 'gemini' (default) or 'ollama'.

    Optional speed knob for Gemini: set GEMINI_THINKING_BUDGET=0 in .env to switch off
    the model's "thinking" phase (faster answers). If your model rejects it, remove the line.
    """
    provider = os.getenv("LLM_PROVIDER", "gemini").lower()
    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(model=os.getenv("OLLAMA_MODEL", "llama3.2"), temperature=0)

    from langchain_google_genai import ChatGoogleGenerativeAI

    kwargs = dict(
        model=os.getenv("GEMINI_MODEL", "gemini-3.8-flash"),
        temperature=0,
        max_retries=2,  # the default retries many times with long waits on rate limits
    )
    budget = os.getenv("GEMINI_THINKING_BUDGET", "").strip()
    if budget:
        kwargs["thinking_budget"] = int(budget)
    return ChatGoogleGenerativeAI(**kwargs)  # reads GOOGLE_API_KEY from the environment


def _to_text(content):
    """Some models return a list of content blocks instead of a plain string."""
    if isinstance(content, str):
        return content
    parts = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "".join(parts)


def retrieve(question, store, k=TOP_K):
    return store.similarity_search(question, k=k)


def build_prompt(question, docs):
    context = "\n\n".join(
        f"[{i}] (source: {d.metadata.get('source_name')}, "
        f"page {d.metadata.get('page_label')})\n{d.page_content}"
        for i, d in enumerate(docs, start=1)
    )
    return PROMPT.format(context=context, question=question)


def answer_stream(question, store, llm, k=TOP_K):
    """Retrieve first (fast), then stream the answer. Returns (docs, text_generator)."""
    docs = retrieve(question, store, k)
    prompt = build_prompt(question, docs)

    def generate():
        for chunk in llm.stream(prompt):
            text = _to_text(chunk.content)
            if text:
                yield text

    return docs, generate()


def answer(question, store, llm, k=TOP_K):
    """Non-streaming version used by evaluate.py. Returns (answer_text, source_docs)."""
    t0 = time.time()
    docs = retrieve(question, store, k)
    t1 = time.time()
    response = llm.invoke(build_prompt(question, docs))
    print(f"retrieval {t1 - t0:.2f}s | LLM {time.time() - t1:.2f}s")
    return _to_text(response.content).strip(), docs