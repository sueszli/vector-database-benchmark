"""This module contains the classes that represent Telegram InlineQueryResult."""
from typing import Final, Optional
from telegram import constants
from telegram._telegramobject import TelegramObject
from telegram._utils.types import JSONDict

class InlineQueryResult(TelegramObject):
    """Baseclass for the InlineQueryResult* classes.

    Objects of this class are comparable in terms of equality. Two objects of this class are
    considered equal, if their :attr:`id` is equal.

    Note:
        All URLs passed in inline query results will be available to end users and therefore must
        be assumed to be *public*.

    Examples:
        :any:`Inline Bot <examples.inlinebot>`

    Args:
        type (:obj:`str`): Type of the result.
        id (:obj:`str`): Unique identifier for this result,
            :tg-const:`telegram.InlineQueryResult.MIN_ID_LENGTH`-
            :tg-const:`telegram.InlineQueryResult.MAX_ID_LENGTH` Bytes.

    Attributes:
        type (:obj:`str`): Type of the result.
        id (:obj:`str`): Unique identifier for this result,
            :tg-const:`telegram.InlineQueryResult.MIN_ID_LENGTH`-
            :tg-const:`telegram.InlineQueryResult.MAX_ID_LENGTH` Bytes.

    """
    __slots__ = ('type', 'id')

    def __init__(self, type: str, id: str, *, api_kwargs: Optional[JSONDict]=None):
        if False:
            while True:
                i = 10
        super().__init__(api_kwargs=api_kwargs)
        self.type: str = type
        self.id: str = str(id)
        self._id_attrs = (self.id,)
        self._freeze()
    MIN_ID_LENGTH: Final[int] = constants.InlineQueryResultLimit.MIN_ID_LENGTH
    ':const:`telegram.constants.InlineQueryResultLimit.MIN_ID_LENGTH`\n\n    .. versionadded:: 20.0\n    '
    MAX_ID_LENGTH: Final[int] = constants.InlineQueryResultLimit.MAX_ID_LENGTH
    ':const:`telegram.constants.InlineQueryResultLimit.MAX_ID_LENGTH`\n\n    .. versionadded:: 20.0\n    '