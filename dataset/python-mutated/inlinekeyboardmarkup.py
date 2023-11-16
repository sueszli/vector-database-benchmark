"""This module contains an object that represents a Telegram InlineKeyboardMarkup."""
from typing import TYPE_CHECKING, Optional, Sequence, Tuple
from telegram._inline.inlinekeyboardbutton import InlineKeyboardButton
from telegram._telegramobject import TelegramObject
from telegram._utils.markup import check_keyboard_type
from telegram._utils.types import JSONDict
if TYPE_CHECKING:
    from telegram import Bot

class InlineKeyboardMarkup(TelegramObject):
    """
    This object represents an inline keyboard that appears right next to the message it belongs to.

    Objects of this class are comparable in terms of equality. Two objects of this class are
    considered equal, if their size of :attr:`inline_keyboard` and all the buttons are equal.

    .. figure:: https://core.telegram.org/file/464001863/110f3/I47qTXAD9Z4.120010/e0        ea04f66357b640ec
        :align: center

        An inline keyboard on a message

    .. seealso::
        An another kind of keyboard would be the :class:`telegram.ReplyKeyboardMarkup`.

    Examples:
        * :any:`Inline Keyboard 1 <examples.inlinekeyboard>`
        * :any:`Inline Keyboard 2 <examples.inlinekeyboard2>`

    Args:
        inline_keyboard (Sequence[Sequence[:class:`telegram.InlineKeyboardButton`]]): Sequence of
            button rows, each represented by a sequence of :class:`~telegram.InlineKeyboardButton`
            objects.

            .. versionchanged:: 20.0
                |sequenceclassargs|

    Attributes:
        inline_keyboard (Tuple[Tuple[:class:`telegram.InlineKeyboardButton`]]): Tuple of
            button rows, each represented by a tuple of :class:`~telegram.InlineKeyboardButton`
            objects.

            .. versionchanged:: 20.0
                |tupleclassattrs|

    """
    __slots__ = ('inline_keyboard',)

    def __init__(self, inline_keyboard: Sequence[Sequence[InlineKeyboardButton]], *, api_kwargs: Optional[JSONDict]=None):
        if False:
            while True:
                i = 10
        super().__init__(api_kwargs=api_kwargs)
        if not check_keyboard_type(inline_keyboard):
            raise ValueError('The parameter `inline_keyboard` should be a sequence of sequences of InlineKeyboardButtons')
        self.inline_keyboard: Tuple[Tuple[InlineKeyboardButton, ...], ...] = tuple((tuple(row) for row in inline_keyboard))
        self._id_attrs = (self.inline_keyboard,)
        self._freeze()

    @classmethod
    def de_json(cls, data: Optional[JSONDict], bot: 'Bot') -> Optional['InlineKeyboardMarkup']:
        if False:
            i = 10
            return i + 15
        'See :meth:`telegram.TelegramObject.de_json`.'
        if not data:
            return None
        keyboard = []
        for row in data['inline_keyboard']:
            tmp = []
            for col in row:
                btn = InlineKeyboardButton.de_json(col, bot)
                if btn:
                    tmp.append(btn)
            keyboard.append(tmp)
        return cls(keyboard)

    @classmethod
    def from_button(cls, button: InlineKeyboardButton, **kwargs: object) -> 'InlineKeyboardMarkup':
        if False:
            while True:
                i = 10
        'Shortcut for::\n\n            InlineKeyboardMarkup([[button]], **kwargs)\n\n        Return an InlineKeyboardMarkup from a single InlineKeyboardButton\n\n        Args:\n            button (:class:`telegram.InlineKeyboardButton`): The button to use in the markup\n\n        '
        return cls([[button]], **kwargs)

    @classmethod
    def from_row(cls, button_row: Sequence[InlineKeyboardButton], **kwargs: object) -> 'InlineKeyboardMarkup':
        if False:
            for i in range(10):
                print('nop')
        'Shortcut for::\n\n            InlineKeyboardMarkup([button_row], **kwargs)\n\n        Return an InlineKeyboardMarkup from a single row of InlineKeyboardButtons\n\n        Args:\n            button_row (Sequence[:class:`telegram.InlineKeyboardButton`]): The button to use\n                in the markup\n\n                .. versionchanged:: 20.0\n                    |sequenceargs|\n\n        '
        return cls([button_row], **kwargs)

    @classmethod
    def from_column(cls, button_column: Sequence[InlineKeyboardButton], **kwargs: object) -> 'InlineKeyboardMarkup':
        if False:
            i = 10
            return i + 15
        'Shortcut for::\n\n            InlineKeyboardMarkup([[button] for button in button_column], **kwargs)\n\n        Return an InlineKeyboardMarkup from a single column of InlineKeyboardButtons\n\n        Args:\n            button_column (Sequence[:class:`telegram.InlineKeyboardButton`]): The button to use\n                in the markup\n\n                 .. versionchanged:: 20.0\n                    |sequenceargs|\n\n        '
        button_grid = [[button] for button in button_column]
        return cls(button_grid, **kwargs)