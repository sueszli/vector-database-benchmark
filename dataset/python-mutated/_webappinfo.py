"""This module contains an object that represents a Telegram Web App Info."""
from typing import Optional
from telegram._telegramobject import TelegramObject
from telegram._utils.types import JSONDict

class WebAppInfo(TelegramObject):
    """
    This object contains information about a `Web App <https://core.telegram.org/bots/webapps>`_.

    Objects of this class are comparable in terms of equality. Two objects of this class are
    considered equal, if their :attr:`url` are equal.

    Examples:
        :any:`Webapp Bot <examples.webappbot>`

    .. versionadded:: 20.0

    Args:
        url (:obj:`str`): An HTTPS URL of a Web App to be opened with additional data as specified
            in `Initializing Web Apps             <https://core.telegram.org/bots/webapps#initializing-mini-apps>`_.

    Attributes:
        url (:obj:`str`): An HTTPS URL of a Web App to be opened with additional data as specified
            in `Initializing Web Apps             <https://core.telegram.org/bots/webapps#initializing-mini-apps>`_.
    """
    __slots__ = ('url',)

    def __init__(self, url: str, *, api_kwargs: Optional[JSONDict]=None):
        if False:
            while True:
                i = 10
        super().__init__(api_kwargs=api_kwargs)
        self.url: str = url
        self._id_attrs = (self.url,)
        self._freeze()