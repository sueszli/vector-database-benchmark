from __future__ import annotations
from typing import TYPE_CHECKING, Any
from .base import TelegramObject

class MessageId(TelegramObject):
    """
    This object represents a unique message identifier.

    Source: https://core.telegram.org/bots/api#messageid
    """
    message_id: int
    'Unique message identifier'
    if TYPE_CHECKING:

        def __init__(__pydantic__self__, *, message_id: int, **__pydantic_kwargs: Any) -> None:
            if False:
                for i in range(10):
                    print('nop')
            super().__init__(message_id=message_id, **__pydantic_kwargs)