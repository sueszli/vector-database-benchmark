"""This module contains an object that represents a Telegram Sent Web App Message."""
from typing import Optional
from telegram._telegramobject import TelegramObject
from telegram._utils.types import JSONDict

class SentWebAppMessage(TelegramObject):
    """Contains information about an inline message sent by a Web App on behalf of a user.

    Objects of this class are comparable in terms of equality. Two objects of this class are
    considered equal, if their :attr:`inline_message_id` are equal.

    .. versionadded:: 20.0

    Args:
        inline_message_id (:obj:`str`, optional): Identifier of the sent inline message. Available
            only if there is an :attr:`inline keyboard <telegram.InlineKeyboardMarkup>` attached to
            the message.

    Attributes:
        inline_message_id (:obj:`str`): Optional. Identifier of the sent inline message. Available
            only if there is an :attr:`inline keyboard <telegram.InlineKeyboardMarkup>` attached to
            the message.
    """
    __slots__ = ('inline_message_id',)

    def __init__(self, inline_message_id: Optional[str]=None, *, api_kwargs: Optional[JSONDict]=None):
        if False:
            while True:
                i = 10
        super().__init__(api_kwargs=api_kwargs)
        self.inline_message_id: Optional[str] = inline_message_id
        self._id_attrs = (self.inline_message_id,)
        self._freeze()