"""This module contains an object that represents a Telegram VideoNote."""
from typing import Optional
from telegram._files._basethumbedmedium import _BaseThumbedMedium
from telegram._files.photosize import PhotoSize
from telegram._utils.types import JSONDict

class VideoNote(_BaseThumbedMedium):
    """This object represents a video message (available in Telegram apps as of v.4.0).

    Objects of this class are comparable in terms of equality. Two objects of this class are
    considered equal, if their :attr:`file_unique_id` is equal.

    .. versionchanged:: 20.5
      |removed_thumb_note|

    Args:
        file_id (:obj:`str`): Identifier for this file, which can be used to download
            or reuse the file.
        file_unique_id (:obj:`str`): Unique identifier for this file, which
            is supposed to be the same over time and for different bots.
            Can't be used to download or reuse the file.
        length (:obj:`int`): Video width and height (diameter of the video message) as defined
            by sender.
        duration (:obj:`int`): Duration of the video in seconds as defined by sender.
        file_size (:obj:`int`, optional): File size in bytes.
        thumbnail (:class:`telegram.PhotoSize`, optional): Video thumbnail.

            .. versionadded:: 20.2

    Attributes:
        file_id (:obj:`str`): Identifier for this file, which can be used to download
            or reuse the file.
        file_unique_id (:obj:`str`): Unique identifier for this file, which
            is supposed to be the same over time and for different bots.
            Can't be used to download or reuse the file.
        length (:obj:`int`): Video width and height (diameter of the video message) as defined
            by sender.
        duration (:obj:`int`): Duration of the video in seconds as defined by sender.
        file_size (:obj:`int`): Optional. File size in bytes.
        thumbnail (:class:`telegram.PhotoSize`): Optional. Video thumbnail.

            .. versionadded:: 20.2

    """
    __slots__ = ('duration', 'length')

    def __init__(self, file_id: str, file_unique_id: str, length: int, duration: int, file_size: Optional[int]=None, thumbnail: Optional[PhotoSize]=None, *, api_kwargs: Optional[JSONDict]=None):
        if False:
            print('Hello World!')
        super().__init__(file_id=file_id, file_unique_id=file_unique_id, file_size=file_size, thumbnail=thumbnail, api_kwargs=api_kwargs)
        with self._unfrozen():
            self.length: int = length
            self.duration: int = duration