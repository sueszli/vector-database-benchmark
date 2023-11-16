"""This module contains an object that represents a Telegram ForceReply."""
from typing import Final, Optional
from telegram import constants
from telegram._telegramobject import TelegramObject
from telegram._utils.types import JSONDict

class ForceReply(TelegramObject):
    """
    Upon receiving a message with this object, Telegram clients will display a reply interface to
    the user (act as if the user has selected the bot's message and tapped 'Reply'). This can be
    extremely useful if you want to create user-friendly step-by-step interfaces without having
    to sacrifice privacy mode.

    Objects of this class are comparable in terms of equality. Two objects of this class are
    considered equal, if their :attr:`selective` is equal.

    .. versionchanged:: 20.0
        The (undocumented) argument ``force_reply`` was removed and instead :attr:`force_reply`
        is now always set to :obj:`True` as expected by the Bot API.

    Args:
        selective (:obj:`bool`, optional): Use this parameter if you want to force reply from
            specific users only. Targets:

            1) Users that are @mentioned in the :attr:`~telegram.Message.text` of the
               :class:`telegram.Message` object.
            2) If the bot's message is a reply (has ``reply_to_message_id``), sender of the
               original message.

        input_field_placeholder (:obj:`str`, optional): The placeholder to be shown in the input
            field when the reply is active;
            :tg-const:`telegram.ForceReply.MIN_INPUT_FIELD_PLACEHOLDER`-
            :tg-const:`telegram.ForceReply.MAX_INPUT_FIELD_PLACEHOLDER`
            characters.

            .. versionadded:: 13.7

    Attributes:
        force_reply (:obj:`True`): Shows reply interface to the user, as if they manually selected
            the bots message and tapped 'Reply'.
        selective (:obj:`bool`): Optional. Force reply from specific users only. Targets:

            1) Users that are @mentioned in the :attr:`~telegram.Message.text` of the
               :class:`telegram.Message` object.
            2) If the bot's message is a reply (has ``reply_to_message_id``), sender of the
               original message.
        input_field_placeholder (:obj:`str`): Optional. The placeholder to be shown in the input
            field when the reply is active;
            :tg-const:`telegram.ForceReply.MIN_INPUT_FIELD_PLACEHOLDER`-
            :tg-const:`telegram.ForceReply.MAX_INPUT_FIELD_PLACEHOLDER`
            characters.

            .. versionadded:: 13.7

    """
    __slots__ = ('selective', 'force_reply', 'input_field_placeholder')

    def __init__(self, selective: Optional[bool]=None, input_field_placeholder: Optional[str]=None, *, api_kwargs: Optional[JSONDict]=None):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(api_kwargs=api_kwargs)
        self.force_reply: bool = True
        self.selective: Optional[bool] = selective
        self.input_field_placeholder: Optional[str] = input_field_placeholder
        self._id_attrs = (self.selective,)
        self._freeze()
    MIN_INPUT_FIELD_PLACEHOLDER: Final[int] = constants.ReplyLimit.MIN_INPUT_FIELD_PLACEHOLDER
    ':const:`telegram.constants.ReplyLimit.MIN_INPUT_FIELD_PLACEHOLDER`\n\n    .. versionadded:: 20.0\n    '
    MAX_INPUT_FIELD_PLACEHOLDER: Final[int] = constants.ReplyLimit.MAX_INPUT_FIELD_PLACEHOLDER
    ':const:`telegram.constants.ReplyLimit.MAX_INPUT_FIELD_PLACEHOLDER`\n\n    .. versionadded:: 20.0\n    '