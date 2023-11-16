"""This module contains an object that represents a Telegram Bot Command."""
from typing import Final, Optional
from telegram import constants
from telegram._telegramobject import TelegramObject
from telegram._utils.types import JSONDict

class BotCommand(TelegramObject):
    """
    This object represents a bot command.

    Objects of this class are comparable in terms of equality. Two objects of this class are
    considered equal, if their :attr:`command` and :attr:`description` are equal.

    Args:
        command (:obj:`str`): Text of the command; :tg-const:`telegram.BotCommand.MIN_COMMAND`-
            :tg-const:`telegram.BotCommand.MAX_COMMAND` characters. Can contain only lowercase
            English letters, digits and underscores.
        description (:obj:`str`): Description of the command;
            :tg-const:`telegram.BotCommand.MIN_DESCRIPTION`-
            :tg-const:`telegram.BotCommand.MAX_DESCRIPTION` characters.

    Attributes:
        command (:obj:`str`): Text of the command; :tg-const:`telegram.BotCommand.MIN_COMMAND`-
            :tg-const:`telegram.BotCommand.MAX_COMMAND` characters. Can contain only lowercase
            English letters, digits and underscores.
        description (:obj:`str`): Description of the command;
            :tg-const:`telegram.BotCommand.MIN_DESCRIPTION`-
            :tg-const:`telegram.BotCommand.MAX_DESCRIPTION` characters.

    """
    __slots__ = ('description', 'command')

    def __init__(self, command: str, description: str, *, api_kwargs: Optional[JSONDict]=None):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(api_kwargs=api_kwargs)
        self.command: str = command
        self.description: str = description
        self._id_attrs = (self.command, self.description)
        self._freeze()
    MIN_COMMAND: Final[int] = constants.BotCommandLimit.MIN_COMMAND
    ':const:`telegram.constants.BotCommandLimit.MIN_COMMAND`\n\n    .. versionadded:: 20.0\n    '
    MAX_COMMAND: Final[int] = constants.BotCommandLimit.MAX_COMMAND
    ':const:`telegram.constants.BotCommandLimit.MAX_COMMAND`\n\n    .. versionadded:: 20.0\n    '
    MIN_DESCRIPTION: Final[int] = constants.BotCommandLimit.MIN_DESCRIPTION
    ':const:`telegram.constants.BotCommandLimit.MIN_DESCRIPTION`\n\n    .. versionadded:: 20.0\n    '
    MAX_DESCRIPTION: Final[int] = constants.BotCommandLimit.MAX_DESCRIPTION
    ':const:`telegram.constants.BotCommandLimit.MAX_DESCRIPTION`\n\n    .. versionadded:: 20.0\n    '