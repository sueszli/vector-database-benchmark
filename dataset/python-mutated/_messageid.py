"""This module contains an object that represents an instance of a Telegram MessageId."""
from typing import Optional
from telegram._telegramobject import TelegramObject
from telegram._utils.types import JSONDict

class MessageId(TelegramObject):
    """This object represents a unique message identifier.

    Objects of this class are comparable in terms of equality. Two objects of this class are
    considered equal, if their :attr:`message_id` is equal.

    Args:
        message_id (:obj:`int`): Unique message identifier.

    Attributes:
        message_id (:obj:`int`): Unique message identifier.
    """
    __slots__ = ('message_id',)

    def __init__(self, message_id: int, *, api_kwargs: Optional[JSONDict]=None):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(api_kwargs=api_kwargs)
        self.message_id: int = message_id
        self._id_attrs = (self.message_id,)
        self._freeze()