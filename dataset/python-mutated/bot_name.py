from typing import TYPE_CHECKING, Any
from .base import TelegramObject

class BotName(TelegramObject):
    """
    This object represents the bot's name.

    Source: https://core.telegram.org/bots/api#botname
    """
    name: str
    "The bot's name"
    if TYPE_CHECKING:

        def __init__(__pydantic__self__, *, name: str, **__pydantic_kwargs: Any) -> None:
            if False:
                print('Hello World!')
            super().__init__(name=name, **__pydantic_kwargs)