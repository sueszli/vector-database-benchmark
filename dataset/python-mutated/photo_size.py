from __future__ import annotations
from typing import TYPE_CHECKING, Any, Optional
from .base import TelegramObject

class PhotoSize(TelegramObject):
    """
    This object represents one size of a photo or a `file <https://core.telegram.org/bots/api#document>`_ / :class:`aiogram.methods.sticker.Sticker` thumbnail.

    Source: https://core.telegram.org/bots/api#photosize
    """
    file_id: str
    'Identifier for this file, which can be used to download or reuse the file'
    file_unique_id: str
    "Unique identifier for this file, which is supposed to be the same over time and for different bots. Can't be used to download or reuse the file."
    width: int
    'Photo width'
    height: int
    'Photo height'
    file_size: Optional[int] = None
    '*Optional*. File size in bytes'
    if TYPE_CHECKING:

        def __init__(__pydantic__self__, *, file_id: str, file_unique_id: str, width: int, height: int, file_size: Optional[int]=None, **__pydantic_kwargs: Any) -> None:
            if False:
                print('Hello World!')
            super().__init__(file_id=file_id, file_unique_id=file_unique_id, width=width, height=height, file_size=file_size, **__pydantic_kwargs)