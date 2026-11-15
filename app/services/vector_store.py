from dataclasses import dataclass
from time import sleep
from typing import Any

from pinecone import Pinecone, ServerlessSpec

from app.core.config import settings


@dataclass(frozen=True)
class RetrievedChunk:
    text: str
    source: str
    document_id: str


class PineconeVectorStore:
    def _client(self) -> Pinecone:
        if settings.pinecone_api_key is None:
            raise RuntimeError("PINECONE_API_KEY is not configured")
        return Pinecone(api_key=settings.pinecone_api_key.get_secret_value())

    def _index(self, *, create_if_missing: bool) -> Any | None:
        client = self._client()
        index_name = settings.pinecone_index_name
        if index_name not in client.list_indexes().names():
            if not create_if_missing:
                return None
            client.create_index(
                name=index_name,
                dimension=settings.embedding_dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud=settings.pinecone_cloud,
                    region=settings.pinecone_region,
                ),
            )
            for _ in range(60):
                description = client.describe_index(index_name)
                if description.status["ready"]:
                    break
                sleep(1)
            else:
                raise TimeoutError(
                    f"Pinecone index {index_name!r} did not become ready"
                )

        description = client.describe_index(index_name)
        if description.dimension != settings.embedding_dimension:
            raise RuntimeError(
                f"Pinecone index dimension is {description.dimension}; "
                f"configured embedding dimension is {settings.embedding_dimension}"
            )
        return client.Index(index_name)

    def upsert_document(
        self,
        *,
        document_id: str,
        filename: str,
        chunks: list[str],
        embeddings: list[list[float]],
    ) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("Each document chunk must have exactly one embedding")
        index = self._index(create_if_missing=True)
        if index is None:
            raise RuntimeError("Pinecone index could not be initialized")
        vectors = [
            {
                "id": f"{document_id}:{chunk_index}",
                "values": embedding,
                "metadata": {
                    "document_id": document_id,
                    "source": filename,
                    "text": chunk,
                    "chunk_index": chunk_index,
                },
            }
            for chunk_index, (chunk, embedding) in enumerate(
                zip(chunks, embeddings, strict=True)
            )
        ]
        index.upsert(vectors=vectors)

    def delete_document(self, document_id: str) -> None:
        index = self._index(create_if_missing=False)
        if index is not None:
            index.delete(filter={"document_id": {"$eq": document_id}})

    def query(self, embedding: list[float], top_k: int) -> list[RetrievedChunk]:
        index = self._index(create_if_missing=False)
        if index is None:
            return []
        result = index.query(vector=embedding, top_k=top_k, include_metadata=True)
        retrieved: list[RetrievedChunk] = []
        for match in result.matches:
            metadata = match.metadata or {}
            text = metadata.get("text")
            source = metadata.get("source")
            document_id = metadata.get("document_id")
            if (
                isinstance(text, str)
                and isinstance(source, str)
                and isinstance(document_id, str)
            ):
                retrieved.append(
                    RetrievedChunk(text=text, source=source, document_id=document_id)
                )
        return retrieved


vector_store = PineconeVectorStore()
