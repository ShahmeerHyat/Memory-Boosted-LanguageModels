from pydantic import BaseModel
from typing import Optional
from enum import Enum


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class Message(BaseModel):
    role: MessageRole
    content: str
    timestamp: Optional[float] = None


class ChatRequest(BaseModel):
    session_id: str
    message: str


class SessionStats(BaseModel):
    session_id: str
    turn_count: int
    retained_turns: int
