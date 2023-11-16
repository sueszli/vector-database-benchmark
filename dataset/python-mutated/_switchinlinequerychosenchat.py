"""This module contains a class that represents a Telegram SwitchInlineQueryChosenChat."""
from typing import Optional
from telegram._telegramobject import TelegramObject
from telegram._utils.types import JSONDict

class SwitchInlineQueryChosenChat(TelegramObject):
    """
    This object represents an inline button that switches the current user to inline mode in a
    chosen chat, with an optional default inline query.

    Objects of this class are comparable in terms of equality. Two objects of this class are
    considered equal, if their :attr:`query`, :attr:`allow_user_chats`, :attr:`allow_bot_chats`,
    :attr:`allow_group_chats`, and :attr:`allow_channel_chats` are equal.

    .. versionadded:: 20.3

    Caution:
        The PTB team has discovered that you must pass at least one of
        :paramref:`allow_user_chats`, :paramref:`allow_bot_chats`, :paramref:`allow_group_chats`,
        or :paramref:`allow_channel_chats` to Telegram. Otherwise, an error will be raised.

    Args:
        query (:obj:`str`, optional): The default inline query to be inserted in the input field.
            If left empty, only the bot's username will be inserted.
        allow_user_chats (:obj:`bool`, optional): Pass :obj:`True`, if private chats with users
            can be chosen.
        allow_bot_chats (:obj:`bool`, optional): Pass :obj:`True`, if private chats with bots can
            be chosen.
        allow_group_chats (:obj:`bool`, optional): Pass :obj:`True`, if group and supergroup chats
            can be chosen.
        allow_channel_chats (:obj:`bool`, optional): Pass :obj:`True`, if channel chats can be
            chosen.

    Attributes:
        query (:obj:`str`): Optional. The default inline query to be inserted in the input field.
            If left empty, only the bot's username will be inserted.
        allow_user_chats (:obj:`bool`): Optional. :obj:`True`, if private chats with users can be
            chosen.
        allow_bot_chats (:obj:`bool`): Optional. :obj:`True`, if private chats with bots can be
            chosen.
        allow_group_chats (:obj:`bool`): Optional. :obj:`True`, if group and supergroup chats can
            be chosen.
        allow_channel_chats (:obj:`bool`): Optional. :obj:`True`, if channel chats can be chosen.

    """
    __slots__ = ('query', 'allow_user_chats', 'allow_bot_chats', 'allow_group_chats', 'allow_channel_chats')

    def __init__(self, query: Optional[str]=None, allow_user_chats: Optional[bool]=None, allow_bot_chats: Optional[bool]=None, allow_group_chats: Optional[bool]=None, allow_channel_chats: Optional[bool]=None, *, api_kwargs: Optional[JSONDict]=None):
        if False:
            print('Hello World!')
        super().__init__(api_kwargs=api_kwargs)
        self.query: Optional[str] = query
        self.allow_user_chats: Optional[bool] = allow_user_chats
        self.allow_bot_chats: Optional[bool] = allow_bot_chats
        self.allow_group_chats: Optional[bool] = allow_group_chats
        self.allow_channel_chats: Optional[bool] = allow_channel_chats
        self._id_attrs = (self.query, self.allow_user_chats, self.allow_bot_chats, self.allow_group_chats, self.allow_channel_chats)
        self._freeze()