from datetime import date, time
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, EmailStr, Field


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=10_000)
    conversation_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=128,
        pattern=r"^[a-zA-Z0-9._:-]+$",
    )

    def resolved_conversation_id(self) -> str:
        return self.conversation_id or str(uuid4())


class ChatResponse(BaseModel):
    conversation_id: str
    answer: str
    booking_saved: bool = False
    booking_id: str | None = None


class BookingDetails(BaseModel):
    name: str | None = None
    email: str | None = None
    date: str | None = None
    time: str | None = None


class AssistantTurn(BaseModel):
    answer: str
    booking_intent: bool = False
    booking: BookingDetails | None = None


class BookingInput(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    email: EmailStr
    date: date
    time: time


class ConversationMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str
