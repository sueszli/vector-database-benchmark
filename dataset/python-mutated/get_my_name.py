from typing import TYPE_CHECKING, Any, Optional
from ..types import BotName
from .base import TelegramMethod

class GetMyName(TelegramMethod[BotName]):
    """
    Use this method to get the current bot name for the given user language. Returns :class:`aiogram.types.bot_name.BotName` on success.

    Source: https://core.telegram.org/bots/api#getmyname
    """
    __returning__ = BotName
    __api_method__ = 'getMyName'
    language_code: Optional[str] = None
    'A two-letter ISO 639-1 language code or an empty string'
    if TYPE_CHECKING:

        def __init__(__pydantic__self__, *, language_code: Optional[str]=None, **__pydantic_kwargs: Any) -> None:
            if False:
                return 10
            super().__init__(language_code=language_code, **__pydantic_kwargs)