"""This module contains the class Defaults, which allows passing default values to Application."""
import datetime
from typing import Any, Dict, NoReturn, Optional, final
from telegram._utils.datetime import UTC

@final
class Defaults:
    """Convenience Class to gather all parameters with a (user defined) default value

    .. seealso:: :wiki:`Architecture Overview <Architecture>`,
        :wiki:`Adding Defaults to Your Bot <Adding-defaults-to-your-bot>`

    .. versionchanged:: 20.0
        Removed the argument and attribute ``timeout``. Specify default timeout behavior for the
        networking backend directly via :class:`telegram.ext.ApplicationBuilder` instead.

    Parameters:
        parse_mode (:obj:`str`, optional): |parse_mode|
        disable_notification (:obj:`bool`, optional): |disable_notification|
        disable_web_page_preview (:obj:`bool`, optional): Disables link previews for links in this
            message.
        allow_sending_without_reply (:obj:`bool`, optional): |allow_sending_without_reply|
        quote (:obj:`bool`, optional): If set to :obj:`True`, the reply is sent as an actual reply
            to the message. If ``reply_to_message_id`` is passed, this parameter will
            be ignored. Default: :obj:`True` in group chats and :obj:`False` in private chats.
        tzinfo (:class:`datetime.tzinfo`, optional): A timezone to be used for all date(time)
            inputs appearing throughout PTB, i.e. if a timezone naive date(time) object is passed
            somewhere, it will be assumed to be in :paramref:`tzinfo`. If the
            :class:`telegram.ext.JobQueue` is used, this must be a timezone provided
            by the ``pytz`` module. Defaults to ``pytz.utc``, if available, and
            :attr:`datetime.timezone.utc` otherwise.
        block (:obj:`bool`, optional): Default setting for the :paramref:`BaseHandler.block`
            parameter
            of handlers and error handlers registered through :meth:`Application.add_handler` and
            :meth:`Application.add_error_handler`. Defaults to :obj:`True`.
        protect_content (:obj:`bool`, optional): |protect_content|

            .. versionadded:: 20.0
    """
    __slots__ = ('_tzinfo', '_disable_web_page_preview', '_block', '_quote', '_disable_notification', '_allow_sending_without_reply', '_parse_mode', '_api_defaults', '_protect_content')

    def __init__(self, parse_mode: Optional[str]=None, disable_notification: Optional[bool]=None, disable_web_page_preview: Optional[bool]=None, quote: Optional[bool]=None, tzinfo: datetime.tzinfo=UTC, block: bool=True, allow_sending_without_reply: Optional[bool]=None, protect_content: Optional[bool]=None):
        if False:
            i = 10
            return i + 15
        self._parse_mode: Optional[str] = parse_mode
        self._disable_notification: Optional[bool] = disable_notification
        self._disable_web_page_preview: Optional[bool] = disable_web_page_preview
        self._allow_sending_without_reply: Optional[bool] = allow_sending_without_reply
        self._quote: Optional[bool] = quote
        self._tzinfo: datetime.tzinfo = tzinfo
        self._block: bool = block
        self._protect_content: Optional[bool] = protect_content
        self._api_defaults = {}
        for kwarg in ('parse_mode', 'explanation_parse_mode', 'disable_notification', 'disable_web_page_preview', 'allow_sending_without_reply', 'protect_content'):
            value = getattr(self, kwarg)
            if value is not None:
                self._api_defaults[kwarg] = value

    def __hash__(self) -> int:
        if False:
            i = 10
            return i + 15
        return hash((self._parse_mode, self._disable_notification, self._disable_web_page_preview, self._allow_sending_without_reply, self._quote, self._tzinfo, self._block, self._protect_content))

    def __eq__(self, other: object) -> bool:
        if False:
            return 10
        if isinstance(other, Defaults):
            return all((getattr(self, attr) == getattr(other, attr) for attr in self.__slots__))
        return False

    @property
    def api_defaults(self) -> Dict[str, Any]:
        if False:
            i = 10
            return i + 15
        return self._api_defaults

    @property
    def parse_mode(self) -> Optional[str]:
        if False:
            while True:
                i = 10
        ":obj:`str`: Optional. Send Markdown or HTML, if you want Telegram apps to show\n        bold, italic, fixed-width text or URLs in your bot's message.\n        "
        return self._parse_mode

    @parse_mode.setter
    def parse_mode(self, value: object) -> NoReturn:
        if False:
            print('Hello World!')
        raise AttributeError('You can not assign a new value to parse_mode after initialization.')

    @property
    def explanation_parse_mode(self) -> Optional[str]:
        if False:
            i = 10
            return i + 15
        ':obj:`str`: Optional. Alias for :attr:`parse_mode`, used for\n        the corresponding parameter of :meth:`telegram.Bot.send_poll`.\n        '
        return self._parse_mode

    @explanation_parse_mode.setter
    def explanation_parse_mode(self, value: object) -> NoReturn:
        if False:
            for i in range(10):
                print('nop')
        raise AttributeError('You can not assign a new value to explanation_parse_mode after initialization.')

    @property
    def disable_notification(self) -> Optional[bool]:
        if False:
            return 10
        ':obj:`bool`: Optional. Sends the message silently. Users will\n        receive a notification with no sound.\n        '
        return self._disable_notification

    @disable_notification.setter
    def disable_notification(self, value: object) -> NoReturn:
        if False:
            for i in range(10):
                print('nop')
        raise AttributeError('You can not assign a new value to disable_notification after initialization.')

    @property
    def disable_web_page_preview(self) -> Optional[bool]:
        if False:
            i = 10
            return i + 15
        ':obj:`bool`: Optional. Disables link previews for links in this\n        message.\n        '
        return self._disable_web_page_preview

    @disable_web_page_preview.setter
    def disable_web_page_preview(self, value: object) -> NoReturn:
        if False:
            return 10
        raise AttributeError('You can not assign a new value to disable_web_page_preview after initialization.')

    @property
    def allow_sending_without_reply(self) -> Optional[bool]:
        if False:
            return 10
        ':obj:`bool`: Optional. Pass :obj:`True`, if the message\n        should be sent even if the specified replied-to message is not found.\n        '
        return self._allow_sending_without_reply

    @allow_sending_without_reply.setter
    def allow_sending_without_reply(self, value: object) -> NoReturn:
        if False:
            for i in range(10):
                print('nop')
        raise AttributeError('You can not assign a new value to allow_sending_without_reply after initialization.')

    @property
    def quote(self) -> Optional[bool]:
        if False:
            return 10
        ':obj:`bool`: Optional. If set to :obj:`True`, the reply is sent as an actual reply\n        to the message. If ``reply_to_message_id`` is passed, this parameter will\n        be ignored. Default: :obj:`True` in group chats and :obj:`False` in private chats.\n        '
        return self._quote

    @quote.setter
    def quote(self, value: object) -> NoReturn:
        if False:
            print('Hello World!')
        raise AttributeError('You can not assign a new value to quote after initialization.')

    @property
    def tzinfo(self) -> datetime.tzinfo:
        if False:
            print('Hello World!')
        ':obj:`tzinfo`: A timezone to be used for all date(time) objects appearing\n        throughout PTB.\n        '
        return self._tzinfo

    @tzinfo.setter
    def tzinfo(self, value: object) -> NoReturn:
        if False:
            while True:
                i = 10
        raise AttributeError('You can not assign a new value to tzinfo after initialization.')

    @property
    def block(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        ':obj:`bool`: Optional. Default setting for the :paramref:`BaseHandler.block` parameter\n        of handlers and error handlers registered through :meth:`Application.add_handler` and\n        :meth:`Application.add_error_handler`.\n        '
        return self._block

    @block.setter
    def block(self, value: object) -> NoReturn:
        if False:
            i = 10
            return i + 15
        raise AttributeError('You can not assign a new value to block after initialization.')

    @property
    def protect_content(self) -> Optional[bool]:
        if False:
            print('Hello World!')
        ':obj:`bool`: Optional. Protects the contents of the sent message from forwarding and\n        saving.\n\n        .. versionadded:: 20.0\n        '
        return self._protect_content

    @protect_content.setter
    def protect_content(self, value: object) -> NoReturn:
        if False:
            return 10
        raise AttributeError("You can't assign a new value to protect_content after initialization.")