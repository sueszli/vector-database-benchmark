"""This module contains an object related to a Telegram Story."""
from typing import Optional
from telegram._telegramobject import TelegramObject
from telegram._utils.types import JSONDict

class Story(TelegramObject):
    """
    This object represents a message about a forwarded story in the chat. Currently holds no
    information.

    .. versionadded:: 20.5

    """
    __slots__ = ()

    def __init__(self, *, api_kwargs: Optional[JSONDict]=None) -> None:
        if False:
            print('Hello World!')
        super().__init__(api_kwargs=api_kwargs)
        self._freeze()