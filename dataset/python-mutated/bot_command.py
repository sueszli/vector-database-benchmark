from __future__ import annotations
from typing import TYPE_CHECKING, Any
from .base import MutableTelegramObject

class BotCommand(MutableTelegramObject):
    """
    This object represents a bot command.

    Source: https://core.telegram.org/bots/api#botcommand
    """
    command: str
    'Text of the command; 1-32 characters. Can contain only lowercase English letters, digits and underscores.'
    description: str
    'Description of the command; 1-256 characters.'
    if TYPE_CHECKING:

        def __init__(__pydantic__self__, *, command: str, description: str, **__pydantic_kwargs: Any) -> None:
            if False:
                while True:
                    i = 10
            super().__init__(command=command, description=description, **__pydantic_kwargs)