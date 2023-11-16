from __future__ import annotations
from typing import TYPE_CHECKING, Any, List
from .base import MutableTelegramObject
if TYPE_CHECKING:
    from .inline_keyboard_button import InlineKeyboardButton

class InlineKeyboardMarkup(MutableTelegramObject):
    """
    This object represents an `inline keyboard <https://core.telegram.org/bots/features#inline-keyboards>`_ that appears right next to the message it belongs to.
    **Note:** This will only work in Telegram versions released after 9 April, 2016. Older clients will display *unsupported message*.

    Source: https://core.telegram.org/bots/api#inlinekeyboardmarkup
    """
    inline_keyboard: List[List[InlineKeyboardButton]]
    'Array of button rows, each represented by an Array of :class:`aiogram.types.inline_keyboard_button.InlineKeyboardButton` objects'
    if TYPE_CHECKING:

        def __init__(__pydantic__self__, *, inline_keyboard: List[List[InlineKeyboardButton]], **__pydantic_kwargs: Any) -> None:
            if False:
                print('Hello World!')
            super().__init__(inline_keyboard=inline_keyboard, **__pydantic_kwargs)