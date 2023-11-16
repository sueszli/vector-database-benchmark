"""This module contains objects related to the write access allowed service message."""
from typing import Optional
from telegram._telegramobject import TelegramObject
from telegram._utils.types import JSONDict

class WriteAccessAllowed(TelegramObject):
    """
    This object represents a service message about a user allowing a bot to write messages after
    adding it to the attachment menu, launching a Web App from a link, or accepting an explicit
    request from a Web App sent by the method
    `requestWriteAccess <https://core.telegram.org/bots/webapps#initializing-mini-apps>`_.

    Objects of this class are comparable in terms of equality. Two objects of this class are
    considered equal, if their :attr:`web_app_name` is equal.

    .. versionadded:: 20.0
    .. versionchanged:: 20.6
       Added custom equality comparison for objects of this class.

    Args:
        web_app_name (:obj:`str`, optional): Name of the Web App, if the access was granted when
            the Web App was launched from a link.

            .. versionadded:: 20.3
        from_request (:obj:`bool`, optional): :obj:`True`, if the access was granted after the user
         accepted an explicit request from a Web App sent by the method
         `requestWriteAccess <https://core.telegram.org/bots/webapps#initializing-mini-apps>`_.

         .. versionadded:: 20.6
        from_attachment_menu (:obj:`bool`, optional): :obj:`True`, if the access was granted when
         the bot was added to the attachment or side menu.

         .. versionadded:: 20.6

    Attributes:
        web_app_name (:obj:`str`): Optional. Name of the Web App, if the access was granted when
            the Web App was launched from a link.

            .. versionadded:: 20.3
        from_request (:obj:`bool`): Optional. :obj:`True`, if the access was granted after the user
            accepted an explicit request from a Web App.

            .. versionadded:: 20.6
        from_attachment_menu (:obj:`bool`): Optional. :obj:`True`, if the access was granted when
            the bot was added to the attachment or side menu.

            .. versionadded:: 20.6

    """
    __slots__ = ('web_app_name', 'from_request', 'from_attachment_menu')

    def __init__(self, web_app_name: Optional[str]=None, from_request: Optional[bool]=None, from_attachment_menu: Optional[bool]=None, *, api_kwargs: Optional[JSONDict]=None):
        if False:
            print('Hello World!')
        super().__init__(api_kwargs=api_kwargs)
        self.web_app_name: Optional[str] = web_app_name
        self.from_request: Optional[bool] = from_request
        self.from_attachment_menu: Optional[bool] = from_attachment_menu
        self._id_attrs = (self.web_app_name,)
        self._freeze()