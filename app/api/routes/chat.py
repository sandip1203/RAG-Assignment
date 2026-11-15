from typing import Annotated

from fastapi import APIRouter, Depends
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.redis import get_redis_client
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.chat_service import answer_message

router = APIRouter(tags=["Chat"])


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    session: Annotated[AsyncSession, Depends(get_db)],
    redis_client: Annotated[Redis, Depends(get_redis_client)],
) -> ChatResponse:
    return await answer_message(
        request=request,
        session=session,
        redis_client=redis_client,
    )
