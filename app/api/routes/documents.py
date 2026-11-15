from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.database import get_db
from app.schemas.documents import ChunkingStrategy, DocumentIngestResponse
from app.services.document_service import ingest_document

router = APIRouter(prefix="/documents", tags=["Documents"])


@router.post(
    "/ingest",
    response_model=DocumentIngestResponse,
    status_code=status.HTTP_201_CREATED,
)
async def ingest(
    session: Annotated[AsyncSession, Depends(get_db)],
    file: Annotated[UploadFile, File()],
    strategy: Annotated[ChunkingStrategy, Form()] = "recursive",
) -> DocumentIngestResponse:
    filename = file.filename or ""
    if filename.rpartition(".")[2].lower() not in {"pdf", "txt"}:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Only .pdf and .txt files are supported",
        )
    content = await file.read(settings.max_upload_size_bytes + 1)
    if len(content) > settings.max_upload_size_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="Uploaded file exceeds the configured size limit",
        )
    try:
        record = await ingest_document(
            session=session,
            filename=filename,
            content=content,
            strategy=strategy,
        )
    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(error),
        ) from error
    return DocumentIngestResponse(
        document_id=record.id,
        filename=record.filename,
        chunking_strategy=strategy,
        chunk_count=record.chunk_count,
    )
