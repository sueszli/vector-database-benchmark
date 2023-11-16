from __future__ import annotations
from typing import TYPE_CHECKING, Any
from .base import TelegramMethod

class SetStickerSetTitle(TelegramMethod[bool]):
    """
    Use this method to set the title of a created sticker set. Returns :code:`True` on success.

    Source: https://core.telegram.org/bots/api#setstickersettitle
    """
    __returning__ = bool
    __api_method__ = 'setStickerSetTitle'
    name: str
    'Sticker set name'
    title: str
    'Sticker set title, 1-64 characters'
    if TYPE_CHECKING:

        def __init__(__pydantic__self__, *, name: str, title: str, **__pydantic_kwargs: Any) -> None:
            if False:
                i = 10
                return i + 15
            super().__init__(name=name, title=title, **__pydantic_kwargs)