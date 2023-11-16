"""This module contains an object that represents a Telegram ReplyKeyboardRemove."""
from typing import Optional
from telegram._telegramobject import TelegramObject
from telegram._utils.types import JSONDict

class ReplyKeyboardRemove(TelegramObject):
    """
    Upon receiving a message with this object, Telegram clients will remove the current custom
    keyboard and display the default letter-keyboard. By default, custom keyboards are displayed
    until a new keyboard is sent by a bot. An exception is made for one-time keyboards that are
    hidden immediately after the user presses a button (see :class:`telegram.ReplyKeyboardMarkup`).

    Note:
        User will not be able to summon this keyboard; if you want to hide the keyboard from
        sight but keep it accessible, use :attr:`telegram.ReplyKeyboardMarkup.one_time_keyboard`.

    Examples:
        * Example usage: A user votes in a poll, bot returns confirmation message in reply to
          the vote and removes the keyboard for that user, while still showing the keyboard with
          poll options to users who haven't voted yet.
        * :any:`Conversation Bot <examples.conversationbot>`
        * :any:`Conversation Bot 2 <examples.conversationbot2>`

    Args:
        selective (:obj:`bool`, optional): Use this parameter if you want to remove the keyboard
            for specific users only. Targets:

            1) Users that are @mentioned in the text of the :class:`telegram.Message` object.
            2) If the bot's message is a reply (has ``reply_to_message_id``), sender of
               the original message.

    Attributes:
        remove_keyboard (:obj:`True`): Requests clients to remove the custom keyboard.
        selective (:obj:`bool`): Optional. Remove the keyboard for specific users only.
            Targets:

            1) Users that are @mentioned in the text of the :class:`telegram.Message` object.
            2) If the bot's message is a reply (has ``reply_to_message_id``), sender of
               the original message.

    """
    __slots__ = ('selective', 'remove_keyboard')

    def __init__(self, selective: Optional[bool]=None, *, api_kwargs: Optional[JSONDict]=None):
        if False:
            print('Hello World!')
        super().__init__(api_kwargs=api_kwargs)
        self.remove_keyboard: bool = True
        self.selective: Optional[bool] = selective
        self._freeze()