import re
from typing import Final

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import settings
from app.schemas.documents import ChunkingStrategy

_SENTENCE_BOUNDARY: Final[re.Pattern[str]] = re.compile(r"(?<=[.!?])\s+|\n+")


def split_text(text: str, strategy: ChunkingStrategy) -> list[str]:
    normalized_text = text.strip()
    if not normalized_text:
        return []

    if strategy == "recursive":
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=min(settings.chunk_overlap, settings.chunk_size - 1),
            add_start_index=True,
        )
        return splitter.split_text(normalized_text)

    return _split_by_sentence(normalized_text)


def _split_by_sentence(text: str) -> list[str]:
    sentences = [
        sentence.strip()
        for sentence in _SENTENCE_BOUNDARY.split(text)
        if sentence.strip()
    ]
    chunks: list[str] = []
    current: list[str] = []

    for sentence in sentences:
        if len(sentence) > settings.chunk_size:
            if current:
                chunks.append(" ".join(current))
                current = []
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=settings.chunk_size,
                chunk_overlap=min(settings.chunk_overlap, settings.chunk_size - 1),
            )
            chunks.extend(splitter.split_text(sentence))
            continue

        candidate = " ".join((*current, sentence))
        if current and len(candidate) > settings.chunk_size:
            chunks.append(" ".join(current))
            overlap: list[str] = []
            for previous_sentence in reversed(current):
                proposed_overlap = [previous_sentence, *overlap]
                if (
                    len(proposed_overlap) > settings.sentence_overlap
                    or len(" ".join((*proposed_overlap, sentence)))
                    > settings.chunk_size
                ):
                    break
                overlap = proposed_overlap
            current = [*overlap, sentence]
        else:
            current.append(sentence)

    if current:
        chunks.append(" ".join(current))
    return chunks
