import asyncio
from uuid import uuid4

import pymupdf as fitz
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.entities import DocumentRecord
from app.schemas.documents import ChunkingStrategy
from app.services.chunking import split_text
from app.services.llm_service import get_embedding_model
from app.services.vector_store import vector_store


def extract_text(filename: str, content: bytes) -> tuple[str, str]:
    suffix = filename.rpartition(".")[2].lower()
    if suffix == "txt":
        try:
            return content.decode("utf-8-sig"), "text/plain"
        except UnicodeDecodeError as error:
            raise ValueError("Text files must use UTF-8 encoding") from error
    if suffix == "pdf":
        try:
            with fitz.open(stream=content, filetype="pdf") as document:
                return "\n".join(
                    page.get_text() for page in document
                ), "application/pdf"
        except (fitz.FileDataError, fitz.EmptyFileError) as error:
            raise ValueError("The uploaded PDF could not be read") from error
    raise ValueError("Only .pdf and .txt files are supported")


async def ingest_document(
    *,
    session: AsyncSession,
    filename: str,
    content: bytes,
    strategy: ChunkingStrategy,
) -> DocumentRecord:
    text, content_type = extract_text(filename, content)
    chunks = split_text(text, strategy)
    if not chunks:
        raise ValueError("The uploaded file contains no extractable text")

    embeddings = await get_embedding_model().aembed_documents(chunks)
    if any(len(embedding) != settings.embedding_dimension for embedding in embeddings):
        raise RuntimeError(
            "Embedding response dimension does not match configured dimension"
        )

    record = DocumentRecord(
        id=str(uuid4()),
        filename=filename,
        content_type=content_type,
        chunking_strategy=strategy,
        chunk_count=len(chunks),
    )
    await asyncio.to_thread(
        vector_store.upsert_document,
        document_id=record.id,
        filename=filename,
        chunks=chunks,
        embeddings=embeddings,
    )
    session.add(record)
    try:
        await session.commit()
    except Exception:
        await session.rollback()
        await asyncio.to_thread(vector_store.delete_document, record.id)
        raise
    await session.refresh(record)
    return record
