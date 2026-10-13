from functools import lru_cache
from pathlib import Path
import fitz
from sentence_transformers import SentenceTransformer
from sqlalchemy import select
from sqlalchemy.orm import Session
from app.config import settings
from app.models import Document, DocumentChunk


@lru_cache(maxsize=1)
def get_embedder() -> SentenceTransformer:
    # First run downloads the model from Hugging Face; subsequent runs use its local cache.
    return SentenceTransformer(settings.embedding_model)


def embed_texts(texts: list[str]) -> list[list[float]]:
    vectors = get_embedder().encode(texts, normalize_embeddings=True)
    return vectors.tolist()


def split_into_chunks(text: str) -> list[str]:
    text = " ".join(text.split())
    if not text:
        return []
    size = settings.chunk_size
    overlap = min(settings.chunk_overlap, size - 1)
    chunks: list[str] = []
    start = 0
    while start < len(text):
        chunk = text[start : start + size].strip()
        if chunk:
            chunks.append(chunk)
        if start + size >= len(text):
            break
        start += size - overlap
    return chunks


def extract_pdf_text(path: str | Path) -> str:
    pages: list[str] = []
    with fitz.open(path) as pdf:
        for page in pdf:
            pages.append(page.get_text())
    return "\n".join(pages).strip()


def ingest_pdf(db: Session, filename: str, path: str | Path) -> dict:
    text = extract_pdf_text(path)
    if not text:
        raise ValueError("No readable text was found in this PDF. Scanned PDFs need OCR first.")
    chunks = split_into_chunks(text)
    embeddings = embed_texts(chunks)
    document = Document(filename=filename)
    db.add(document)
    db.flush()
    db.add_all(
        [
            DocumentChunk(
                document_id=document.id,
                chunk_index=index,
                content=chunk,
                embedding=embedding,
            )
            for index, (chunk, embedding) in enumerate(zip(chunks, embeddings))
        ]
    )
    db.commit()
    return {"document_id": document.id, "filename": filename, "chunks_created": len(chunks)}


def search_documents(db: Session, query: str, top_k: int | None = None) -> list[dict]:
    query_embedding = embed_texts([query])[0]
    count = top_k or settings.top_k
    distance = DocumentChunk.embedding.cosine_distance(query_embedding)
    rows = db.execute(
        select(DocumentChunk, Document.filename, distance.label("distance"))
        .join(Document, Document.id == DocumentChunk.document_id)
        .order_by(distance)
        .limit(count)
    ).all()
    return [
        {
            "chunk_id": chunk.id,
            "document_id": chunk.document_id,
            "filename": filename,
            "chunk_index": chunk.chunk_index,
            "content": chunk.content,
            "distance": float(distance_value),
        }
        for chunk, filename, distance_value in rows
    ]
