"""This module contains an object that represents a type of a Telegram Poll."""
from typing import Optional
from telegram._telegramobject import TelegramObject
from telegram._utils.types import JSONDict

class KeyboardButtonPollType(TelegramObject):
    """This object represents type of a poll, which is allowed to be created
    and sent when the corresponding button is pressed.

    Objects of this class are comparable in terms of equality. Two objects of this class are
    considered equal, if their :attr:`type` is equal.

    Examples:
        :any:`Poll Bot <examples.pollbot>`

    Args:
        type (:obj:`str`, optional): If :tg-const:`telegram.Poll.QUIZ` is passed, the user will be
            allowed to create only polls in the quiz mode. If :tg-const:`telegram.Poll.REGULAR` is
            passed, only regular polls will be allowed. Otherwise, the user will be allowed to
            create a poll of any type.
    Attributes:
        type (:obj:`str`): Optional. If equals :tg-const:`telegram.Poll.QUIZ`, the user will
            be allowed to create only polls in the quiz mode. If equals
            :tg-const:`telegram.Poll.REGULAR`, only regular polls will be allowed.
            Otherwise, the user will be allowed to create a poll of any type.
    """
    __slots__ = ('type',)

    def __init__(self, type: Optional[str]=None, *, api_kwargs: Optional[JSONDict]=None):
        if False:
            return 10
        super().__init__(api_kwargs=api_kwargs)
        self.type: Optional[str] = type
        self._id_attrs = (self.type,)
        self._freeze()