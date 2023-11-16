from __future__ import annotations
from typing import TYPE_CHECKING, Any, List, Optional
from .base import TelegramMethod

class SetStickerKeywords(TelegramMethod[bool]):
    """
    Use this method to change search keywords assigned to a regular or custom emoji sticker. The sticker must belong to a sticker set created by the bot. Returns :code:`True` on success.

    Source: https://core.telegram.org/bots/api#setstickerkeywords
    """
    __returning__ = bool
    __api_method__ = 'setStickerKeywords'
    sticker: str
    'File identifier of the sticker'
    keywords: Optional[List[str]] = None
    'A JSON-serialized list of 0-20 search keywords for the sticker with total length of up to 64 characters'
    if TYPE_CHECKING:

        def __init__(__pydantic__self__, *, sticker: str, keywords: Optional[List[str]]=None, **__pydantic_kwargs: Any) -> None:
            if False:
                while True:
                    i = 10
            super().__init__(sticker=sticker, keywords=keywords, **__pydantic_kwargs)