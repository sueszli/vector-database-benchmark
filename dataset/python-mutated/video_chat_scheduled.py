from __future__ import annotations
from typing import TYPE_CHECKING, Any
from .base import TelegramObject
from .custom import DateTime

class VideoChatScheduled(TelegramObject):
    """
    This object represents a service message about a video chat scheduled in the chat.

    Source: https://core.telegram.org/bots/api#videochatscheduled
    """
    start_date: DateTime
    'Point in time (Unix timestamp) when the video chat is supposed to be started by a chat administrator'
    if TYPE_CHECKING:

        def __init__(__pydantic__self__, *, start_date: DateTime, **__pydantic_kwargs: Any) -> None:
            if False:
                i = 10
                return i + 15
            super().__init__(start_date=start_date, **__pydantic_kwargs)