from typing import TYPE_CHECKING, Any
from aiogram.types import TelegramObject

class BotDescription(TelegramObject):
    """
    This object represents the bot's description.

    Source: https://core.telegram.org/bots/api#botdescription
    """
    description: str
    "The bot's description"
    if TYPE_CHECKING:

        def __init__(__pydantic__self__, *, description: str, **__pydantic_kwargs: Any) -> None:
            if False:
                print('Hello World!')
            super().__init__(description=description, **__pydantic_kwargs)