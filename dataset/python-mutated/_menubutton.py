"""This module contains objects related to Telegram menu buttons."""
from typing import TYPE_CHECKING, Dict, Final, Optional, Type
from telegram import constants
from telegram._telegramobject import TelegramObject
from telegram._utils.types import JSONDict
from telegram._webappinfo import WebAppInfo
if TYPE_CHECKING:
    from telegram import Bot

class MenuButton(TelegramObject):
    """This object describes the bot's menu button in a private chat. It should be one of

    * :class:`telegram.MenuButtonCommands`
    * :class:`telegram.MenuButtonWebApp`
    * :class:`telegram.MenuButtonDefault`

    If a menu button other than :class:`telegram.MenuButtonDefault` is set for a private chat,
    then it is applied in the chat. Otherwise the default menu button is applied. By default, the
    menu button opens the list of bot commands.

    Objects of this class are comparable in terms of equality. Two objects of this class are
    considered equal, if their :attr:`type` is equal. For subclasses with additional attributes,
    the notion of equality is overridden.

    .. versionadded:: 20.0

    Args:
        type (:obj:`str`): Type of menu button that the instance represents.

    Attributes:
        type (:obj:`str`): Type of menu button that the instance represents.
    """
    __slots__ = ('type',)

    def __init__(self, type: str, *, api_kwargs: Optional[JSONDict]=None):
        if False:
            print('Hello World!')
        super().__init__(api_kwargs=api_kwargs)
        self.type: str = type
        self._id_attrs = (self.type,)
        self._freeze()

    @classmethod
    def de_json(cls, data: Optional[JSONDict], bot: 'Bot') -> Optional['MenuButton']:
        if False:
            i = 10
            return i + 15
        'Converts JSON data to the appropriate :class:`MenuButton` object, i.e. takes\n        care of selecting the correct subclass.\n\n        Args:\n            data (Dict[:obj:`str`, ...]): The JSON data.\n            bot (:class:`telegram.Bot`): The bot associated with this object.\n\n        Returns:\n            The Telegram object.\n\n        '
        data = cls._parse_data(data)
        if data is None:
            return None
        if not data and cls is MenuButton:
            return None
        _class_mapping: Dict[str, Type[MenuButton]] = {cls.COMMANDS: MenuButtonCommands, cls.WEB_APP: MenuButtonWebApp, cls.DEFAULT: MenuButtonDefault}
        if cls is MenuButton and data.get('type') in _class_mapping:
            return _class_mapping[data.pop('type')].de_json(data, bot=bot)
        return super().de_json(data=data, bot=bot)
    COMMANDS: Final[str] = constants.MenuButtonType.COMMANDS
    ':const:`telegram.constants.MenuButtonType.COMMANDS`'
    WEB_APP: Final[str] = constants.MenuButtonType.WEB_APP
    ':const:`telegram.constants.MenuButtonType.WEB_APP`'
    DEFAULT: Final[str] = constants.MenuButtonType.DEFAULT
    ':const:`telegram.constants.MenuButtonType.DEFAULT`'

class MenuButtonCommands(MenuButton):
    """Represents a menu button, which opens the bot's list of commands.

    .. include:: inclusions/menu_button_command_video.rst

    .. versionadded:: 20.0
    Attributes:
        type (:obj:`str`): :tg-const:`telegram.constants.MenuButtonType.COMMANDS`.
    """
    __slots__ = ()

    def __init__(self, *, api_kwargs: Optional[JSONDict]=None):
        if False:
            while True:
                i = 10
        super().__init__(type=constants.MenuButtonType.COMMANDS, api_kwargs=api_kwargs)
        self._freeze()

class MenuButtonWebApp(MenuButton):
    """Represents a menu button, which launches a
    `Web App <https://core.telegram.org/bots/webapps>`_.

    Objects of this class are comparable in terms of equality. Two objects of this class are
    considered equal, if their :attr:`type`, :attr:`text` and :attr:`web_app`
    are equal.

    .. versionadded:: 20.0

    Args:
        text (:obj:`str`): Text of the button.
        web_app (:class:`telegram.WebAppInfo`): Description of the Web App that will be launched
            when the user presses the button. The Web App will be able to send an arbitrary
            message on behalf of the user using the method :meth:`~telegram.Bot.answerWebAppQuery`
            of :class:`~telegram.Bot`.

    Attributes:
        type (:obj:`str`): :tg-const:`telegram.constants.MenuButtonType.WEB_APP`.
        text (:obj:`str`): Text of the button.
        web_app (:class:`telegram.WebAppInfo`): Description of the Web App that will be launched
            when the user presses the button. The Web App will be able to send an arbitrary
            message on behalf of the user using the method :meth:`~telegram.Bot.answerWebAppQuery`
            of :class:`~telegram.Bot`.
    """
    __slots__ = ('text', 'web_app')

    def __init__(self, text: str, web_app: WebAppInfo, *, api_kwargs: Optional[JSONDict]=None):
        if False:
            while True:
                i = 10
        super().__init__(type=constants.MenuButtonType.WEB_APP, api_kwargs=api_kwargs)
        with self._unfrozen():
            self.text: str = text
            self.web_app: WebAppInfo = web_app
            self._id_attrs = (self.type, self.text, self.web_app)

    @classmethod
    def de_json(cls, data: Optional[JSONDict], bot: 'Bot') -> Optional['MenuButtonWebApp']:
        if False:
            return 10
        'See :meth:`telegram.TelegramObject.de_json`.'
        data = cls._parse_data(data)
        if not data:
            return None
        data['web_app'] = WebAppInfo.de_json(data.get('web_app'), bot)
        return super().de_json(data=data, bot=bot)

class MenuButtonDefault(MenuButton):
    """Describes that no specific value for the menu button was set.

    .. versionadded:: 20.0
    Attributes:
        type (:obj:`str`): :tg-const:`telegram.constants.MenuButtonType.DEFAULT`.
    """
    __slots__ = ()

    def __init__(self, *, api_kwargs: Optional[JSONDict]=None):
        if False:
            while True:
                i = 10
        super().__init__(type=constants.MenuButtonType.DEFAULT, api_kwargs=api_kwargs)
        self._freeze()