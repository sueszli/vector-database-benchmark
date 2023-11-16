"""This module contains two objects that represent a Telegram bots (short) description."""
from typing import Optional
from telegram._telegramobject import TelegramObject
from telegram._utils.types import JSONDict

class BotDescription(TelegramObject):
    """This object represents the bot's description.

    Objects of this class are comparable in terms of equality. Two objects of this class are
    considered equal, if their :attr:`description` is equal.

    .. versionadded:: 20.2

    Args:
        description (:obj:`str`): The bot's description.

    Attributes:
        description (:obj:`str`): The bot's description.

    """
    __slots__ = ('description',)

    def __init__(self, description: str, *, api_kwargs: Optional[JSONDict]=None):
        if False:
            while True:
                i = 10
        super().__init__(api_kwargs=api_kwargs)
        self.description: str = description
        self._id_attrs = (self.description,)
        self._freeze()

class BotShortDescription(TelegramObject):
    """This object represents the bot's short description.

    Objects of this class are comparable in terms of equality. Two objects of this class are
    considered equal, if their :attr:`short_description` is equal.

    .. versionadded:: 20.2

    Args:
        short_description (:obj:`str`): The bot's short description.

    Attributes:
        short_description (:obj:`str`): The bot's short description.

    """
    __slots__ = ('short_description',)

    def __init__(self, short_description: str, *, api_kwargs: Optional[JSONDict]=None):
        if False:
            print('Hello World!')
        super().__init__(api_kwargs=api_kwargs)
        self.short_description: str = short_description
        self._id_attrs = (self.short_description,)
        self._freeze()