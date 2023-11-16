"""This module contains an object that represents a Telegram CallbackGame."""
from typing import Optional
from telegram._telegramobject import TelegramObject
from telegram._utils.types import JSONDict

class CallbackGame(TelegramObject):
    """A placeholder, currently holds no information. Use BotFather to set up your game."""
    __slots__ = ()

    def __init__(self, *, api_kwargs: Optional[JSONDict]=None) -> None:
        if False:
            while True:
                i = 10
        super().__init__(api_kwargs=api_kwargs)
        self._freeze()