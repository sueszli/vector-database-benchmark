"""This module contains an object that represents a Telegram PhotoSize."""
from typing import Optional
from telegram._files._basemedium import _BaseMedium
from telegram._utils.types import JSONDict

class PhotoSize(_BaseMedium):
    """This object represents one size of a photo or a file/sticker thumbnail.

    Objects of this class are comparable in terms of equality. Two objects of this class are
    considered equal, if their :attr:`file_unique_id` is equal.

    Args:
        file_id (:obj:`str`): Identifier for this file, which can be used to download
            or reuse the file.
        file_unique_id (:obj:`str`): Unique identifier for this file, which
            is supposed to be the same over time and for different bots.
            Can't be used to download or reuse the file.
        width (:obj:`int`): Photo width.
        height (:obj:`int`): Photo height.
        file_size (:obj:`int`, optional): File size in bytes.

    Attributes:
        file_id (:obj:`str`): Identifier for this file, which can be used to download
            or reuse the file.
        file_unique_id (:obj:`str`): Unique identifier for this file, which
            is supposed to be the same over time and for different bots.
            Can't be used to download or reuse the file.
        width (:obj:`int`): Photo width.
        height (:obj:`int`): Photo height.
        file_size (:obj:`int`): Optional. File size in bytes.


    """
    __slots__ = ('width', 'height')

    def __init__(self, file_id: str, file_unique_id: str, width: int, height: int, file_size: Optional[int]=None, *, api_kwargs: Optional[JSONDict]=None):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(file_id=file_id, file_unique_id=file_unique_id, file_size=file_size, api_kwargs=api_kwargs)
        with self._unfrozen():
            self.width: int = width
            self.height: int = height