from __future__ import annotations
from typing import TYPE_CHECKING, Any
from .base import TelegramMethod

class DeleteStickerSet(TelegramMethod[bool]):
    """
    Use this method to delete a sticker set that was created by the bot. Returns :code:`True` on success.

    Source: https://core.telegram.org/bots/api#deletestickerset
    """
    __returning__ = bool
    __api_method__ = 'deleteStickerSet'
    name: str
    'Sticker set name'
    if TYPE_CHECKING:

        def __init__(__pydantic__self__, *, name: str, **__pydantic_kwargs: Any) -> None:
            if False:
                while True:
                    i = 10
            super().__init__(name=name, **__pydantic_kwargs)