from __future__ import annotations
from typing import TYPE_CHECKING, Any
from .base import TelegramObject

class VideoChatEnded(TelegramObject):
    """
    This object represents a service message about a video chat ended in the chat.

    Source: https://core.telegram.org/bots/api#videochatended
    """
    duration: int
    'Video chat duration in seconds'
    if TYPE_CHECKING:

        def __init__(__pydantic__self__, *, duration: int, **__pydantic_kwargs: Any) -> None:
            if False:
                while True:
                    i = 10
            super().__init__(duration=duration, **__pydantic_kwargs)