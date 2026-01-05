from __future__ import annotations
from pathlib import Path
from typing import Iterable, List, Optional

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM


def load_pdfs(pdf_paths: Iterable[Path]):
    docs = []
    for p in pdf_paths:
        loader = PyPDFLoader(str(p))
        docs.extend(loader.load())
    return docs


def build_vectorstore(
    pdf_paths: List[Path],
    persist_dir: Path,
    embed_model: str = "embeddinggemma",
    chunk_size: int = 1500,
    chunk_overlap: int = 200,
) -> int:
    raw_docs = load_pdfs(pdf_paths)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_documents(raw_docs)

    emb = OllamaEmbeddings(model=embed_model)
    Chroma.from_documents(
        documents=chunks,
        embedding=emb,
        persist_directory=str(persist_dir),
    )
    return len(chunks)


def answer_query(
    question: str,
    persist_dir: Path,
    embed_model: str = "embeddinggemma",
    chat_model: str = "llama3.1",
    top_k: int = 4,
) -> str:
    emb = OllamaEmbeddings(model=embed_model)
    db = Chroma(persist_directory=str(persist_dir), embedding_function=emb)

    hits = db.similarity_search(question, k=top_k)
    context = "\n\n---\n\n".join([h.page_content for h in hits])

    llm = OllamaLLM(model=chat_model)
    prompt = f"""Bạn chỉ trả lời dựa trên NGỮ CẢNH. Nếu thiếu thông tin thì nói "không đủ thông tin".

NGỮ CẢNH:
{context}

CÂU HỎI: {question}
TRẢ LỜI:"""
    return llm.invoke(prompt)
