"""This module contains an object that represents a change in the Telegram message auto
deletion.
"""
from typing import Optional
from telegram._telegramobject import TelegramObject
from telegram._utils.types import JSONDict

class MessageAutoDeleteTimerChanged(TelegramObject):
    """This object represents a service message about a change in auto-delete timer settings.

    Objects of this class are comparable in terms of equality. Two objects of this class are
    considered equal, if their :attr:`message_auto_delete_time` is equal.

    .. versionadded:: 13.4

    Args:
        message_auto_delete_time (:obj:`int`): New auto-delete time for messages in the
            chat.

    Attributes:
        message_auto_delete_time (:obj:`int`): New auto-delete time for messages in the
            chat.

    """
    __slots__ = ('message_auto_delete_time',)

    def __init__(self, message_auto_delete_time: int, *, api_kwargs: Optional[JSONDict]=None):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(api_kwargs=api_kwargs)
        self.message_auto_delete_time: int = message_auto_delete_time
        self._id_attrs = (self.message_auto_delete_time,)
        self._freeze()