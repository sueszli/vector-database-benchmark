"""This module contains an object that represents a Telegram WebAppData."""
from typing import Optional
from telegram._telegramobject import TelegramObject
from telegram._utils.types import JSONDict

class WebAppData(TelegramObject):
    """Contains data sent from a `Web App <https://core.telegram.org/bots/webapps>`_ to the bot.

    Objects of this class are comparable in terms of equality. Two objects of this class are
    considered equal, if their :attr:`data` and :attr:`button_text` are equal.

    Examples:
        :any:`Webapp Bot <examples.webappbot>`

    .. versionadded:: 20.0

    Args:
        data (:obj:`str`): The data. Be aware that a bad client can send arbitrary data in this
            field.
        button_text (:obj:`str`): Text of the :paramref:`~telegram.KeyboardButton.web_app` keyboard
            button, from which the Web App was opened.

    Attributes:
        data (:obj:`str`): The data. Be aware that a bad client can send arbitrary data in this
            field.
        button_text (:obj:`str`): Text of the :paramref:`~telegram.KeyboardButton.web_app` keyboard
            button, from which the Web App was opened.

            Warning:
                Be aware that a bad client can send arbitrary data in this field.
    """
    __slots__ = ('data', 'button_text')

    def __init__(self, data: str, button_text: str, *, api_kwargs: Optional[JSONDict]=None):
        if False:
            return 10
        super().__init__(api_kwargs=api_kwargs)
        self.data: str = data
        self.button_text: str = button_text
        self._id_attrs = (self.data, self.button_text)
        self._freeze()