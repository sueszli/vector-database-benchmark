from __future__ import annotations
from typing import TYPE_CHECKING, Any
from .base import TelegramObject

class MessageAutoDeleteTimerChanged(TelegramObject):
    """
    This object represents a service message about a change in auto-delete timer settings.

    Source: https://core.telegram.org/bots/api#messageautodeletetimerchanged
    """
    message_auto_delete_time: int
    'New auto-delete time for messages in the chat; in seconds'
    if TYPE_CHECKING:

        def __init__(__pydantic__self__, *, message_auto_delete_time: int, **__pydantic_kwargs: Any) -> None:
            if False:
                return 10
            super().__init__(message_auto_delete_time=message_auto_delete_time, **__pydantic_kwargs)