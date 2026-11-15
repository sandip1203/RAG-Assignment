from typing import Literal

from pydantic import BaseModel, ConfigDict

ChunkingStrategy = Literal["recursive", "sentence"]


class DocumentIngestResponse(BaseModel):
    document_id: str
    filename: str
    chunking_strategy: ChunkingStrategy
    chunk_count: int

    model_config = ConfigDict(from_attributes=True)
