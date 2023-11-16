from typing import TYPE_CHECKING, Any, Optional
from .base import TelegramMethod

class SetMyName(TelegramMethod[bool]):
    """
    Use this method to change the bot's name. Returns :code:`True` on success.

    Source: https://core.telegram.org/bots/api#setmyname
    """
    __returning__ = bool
    __api_method__ = 'setMyName'
    name: Optional[str] = None
    'New bot name; 0-64 characters. Pass an empty string to remove the dedicated name for the given language.'
    language_code: Optional[str] = None
    'A two-letter ISO 639-1 language code. If empty, the name will be shown to all users for whose language there is no dedicated name.'
    if TYPE_CHECKING:

        def __init__(__pydantic__self__, *, name: Optional[str]=None, language_code: Optional[str]=None, **__pydantic_kwargs: Any) -> None:
            if False:
                for i in range(10):
                    print('nop')
            super().__init__(name=name, language_code=language_code, **__pydantic_kwargs)