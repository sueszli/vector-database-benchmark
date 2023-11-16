"""This module contains an object that represent a Telegram bots name."""
from typing import Final, Optional
from telegram import constants
from telegram._telegramobject import TelegramObject
from telegram._utils.types import JSONDict

class BotName(TelegramObject):
    """This object represents the bot's name.

    Objects of this class are comparable in terms of equality. Two objects of this class are
    considered equal, if their :attr:`name` is equal.

    .. versionadded:: 20.3

    Args:
        name (:obj:`str`): The bot's name.

    Attributes:
        name (:obj:`str`): The bot's name.

    """
    __slots__ = ('name',)

    def __init__(self, name: str, *, api_kwargs: Optional[JSONDict]=None):
        if False:
            i = 10
            return i + 15
        super().__init__(api_kwargs=api_kwargs)
        self.name: str = name
        self._id_attrs = (self.name,)
        self._freeze()
    MAX_LENGTH: Final[int] = constants.BotNameLimit.MAX_NAME_LENGTH
    ':const:`telegram.constants.BotNameLimit.MAX_NAME_LENGTH`'