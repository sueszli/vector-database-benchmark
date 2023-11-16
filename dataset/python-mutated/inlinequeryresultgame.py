"""This module contains the classes that represent Telegram InlineQueryResultGame."""
from typing import Optional
from telegram._inline.inlinekeyboardmarkup import InlineKeyboardMarkup
from telegram._inline.inlinequeryresult import InlineQueryResult
from telegram._utils.types import JSONDict
from telegram.constants import InlineQueryResultType

class InlineQueryResultGame(InlineQueryResult):
    """Represents a :class:`telegram.Game`.

    Args:
        id (:obj:`str`): Unique identifier for this result,
            :tg-const:`telegram.InlineQueryResult.MIN_ID_LENGTH`-
            :tg-const:`telegram.InlineQueryResult.MAX_ID_LENGTH` Bytes.
        game_short_name (:obj:`str`): Short name of the game.
        reply_markup (:class:`telegram.InlineKeyboardMarkup`, optional): Inline keyboard attached
            to the message.

    Attributes:
        type (:obj:`str`): :tg-const:`telegram.constants.InlineQueryResultType.GAME`.
        id (:obj:`str`): Unique identifier for this result,
            :tg-const:`telegram.InlineQueryResult.MIN_ID_LENGTH`-
            :tg-const:`telegram.InlineQueryResult.MAX_ID_LENGTH` Bytes.
        game_short_name (:obj:`str`): Short name of the game.
        reply_markup (:class:`telegram.InlineKeyboardMarkup`): Optional. Inline keyboard attached
            to the message.

    """
    __slots__ = ('reply_markup', 'game_short_name')

    def __init__(self, id: str, game_short_name: str, reply_markup: Optional[InlineKeyboardMarkup]=None, *, api_kwargs: Optional[JSONDict]=None):
        if False:
            while True:
                i = 10
        super().__init__(InlineQueryResultType.GAME, id, api_kwargs=api_kwargs)
        with self._unfrozen():
            self.id: str = id
            self.game_short_name: str = game_short_name
            self.reply_markup: Optional[InlineKeyboardMarkup] = reply_markup