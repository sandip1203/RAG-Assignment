from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.router import api_router
from app.core.config import settings
from app.core.database import close_database, initialize_database
from app.core.redis import close_redis


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    await initialize_database()

    yield

    await close_redis()
    await close_database()


app = FastAPI(
    title=settings.app_name,
    description="Document-based Conversational RAG and Interview Booking API",
    version="0.1.0",
    lifespan=lifespan,
)


app.include_router(
    api_router,
    prefix=settings.api_v1_prefix,
)


@app.get("/")
async def root() -> dict[str, str]:
    return {
        "message": "Welcome to RAG APP",
        "docs": "/docs",
        "health": f"{settings.api_v1_prefix}/health",
    }
